import datasets
import transformers
from transformers import AutoTokenizer
from datasets import concatenate_datasets
from http import HTTPStatus
import dashscope
from itertools import chain
from modelzipper.tutils import *
from dashscope import Generation 
import threading


dashscope.api_key="sk-50a14a4626cb44d3a5cd33fff9750f39"

def process_slimpajama():
    dataset = datasets.load_dataset("/vepfs/wcf/G/zecheng/data/SlimPajama-6B", num_proc=12, cache_dir="/vepfs/wcf/G/zecheng/cache")

    tokenizer = AutoTokenizer.from_pretrained("/vepfs/wcf/G/zecheng/hf_models/Mistral-7B-Instruct-v0.2")

    def filter_function(example):
        input_ids = example['text']
        tok_input_ids = tokenizer(input_ids).input_ids
        return 4096 <= len(tok_input_ids) <= 9012

    filtered_train_dataset = dataset['train'].filter(filter_function, num_proc=120)
    filtered_validation_dataset = dataset['validation'].filter(filter_function, num_proc=64)
    filtered_test_dataset = dataset['test'].filter(filter_function, num_proc=64)

    all_filtered_datasets = concatenate_datasets([filtered_train_dataset, filtered_validation_dataset, filtered_test_dataset])

    all_filtered_datasets = all_filtered_datasets.remove_columns(['meta', '__index_level_0__'])

    all_filtered_datasets.save_to_disk("/vepfs/wcf/G/zecheng/data/SlimPajama-6B/processed")


def split_text_into_chunks(text, chunk_size=1024):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text) - chunk_size + 1, chunk_size)]
    return chunks


def process_dataset(dataset):
    def apply_chunking(examples):
        chunks = split_text_into_chunks(examples['text'])
        return {"text": chunks}
    chunked_dataset = dataset.map(apply_chunking, batched=False, num_proc=64)
    chunked_dataset = chunked_dataset.flatten()
    chunked_datasets = list(chain.from_iterable(item['text'] for item in chunked_dataset))
    chunked_datasets = [item['text'] for item in chunked_dataset]
    return chunked_datasets


def load_slimpajama():
    dataset = datasets.load_from_disk("/vepfs/wcf/G/zecheng/data/SlimPajama-6B/processed")
    return dataset


def generate_QA_pairs(dataset):
    dataset = [item['text'] for item in dataset]
    

def call_with_messages(reference_text, model_name):
    SYSTEM_MESSAGE = """
    I will provide you with a snippet of text(reference), and ask you to ask two sets of question-answer pairs based on that snippet of text, each consisting of a question and a response based on the snippet of text(reference), respectively. 
    
    You should determine whether the input content is English input, If there is a lot of code, gibberish, math symbols or Http URLs etc, alive is not suitable for questions and answers, direct reply to an English word: Null 
    
    If it is suitable for the reply, The final format requested to be returned is as follows:
    
    ####Question 1####{Question}####Answer 1####{Answer}####Question 2###{Question}####Answer 2####{Answer}###
    
    Where {Question} is a question generated from the provided text snippet(reference) and {Answer} is a response generated from the text snippet and {Question}.
    """
    
    USER_TEMPLATE = f"Here is the reference text: {reference_text}, please provide two sets of question-answer pairs based on the reference text, must follow the format."

    messages = [{'role': 'system', 'content': SYSTEM_MESSAGE},
                {'role': 'user', 'content': USER_TEMPLATE}]
    
    max_attempts = 2
    attempts = 0

    while attempts < max_attempts:
        response = dashscope.Generation.call(
            model=model_name,
            messages=messages,
            result_format='message', 
        )
        if response.status_code == HTTPStatus.OK:
            return response
        elif response.code == "DataInspectionFailed":
            return "DataInspectionFailed"
        else:
            print('Attempt {0}: Request id: {1}, Status code: {2}, error code: {3}, error message: {4}'.format(
                attempts + 1, response.request_id, response.status_code,
                response.code, response.message
            ))
            time.sleep(10)  # sleep for 10 seconds before retrying
            attempts += 1
    print("Failed after maximum attempts.")
    return None 


def process_chunk(chunk, chunk_length, thread_id):
    output_file = f"/vepfs/wcf/G/zecheng/data/SlimPajama-6B/generated_QA_pairs_thread_{thread_id}.jsonl"
    log_file = f"/vepfs/wcf/G/zecheng/data/SlimPajama-6B/processing_log_thread_{thread_id}.log"
    model_name, all_chunks = chunk[0], chunk[1]
    with open(output_file, 'a') as f, open(log_file, 'a') as log_f:
        with tqdm(total=chunk_length, desc=f"Processing thread {thread_id}") as pbar:
            for sample in all_chunks:
                response = call_with_messages(sample[1], model_name)
                if response is None:
                    continue
                elif response == "DataInspectionFailed":  # has toxic content
                    log_f.write(f"{sample[0]}\n")
                    log_f.flush()
                else:  # has response
                    response = response['output']['choices'][0]['message']['content']
                    if "null" in response.lower():
                        continue
                    pairs = {"reference": sample[1], "QA_pairs": response}
                    json_str = json.dumps(pairs)
                    # Write to output file
                    f.write(json_str + '\n')
                    f.flush()
                    # Write to log file
                    log_f.write(f"{sample[0]}\n")
                    log_f.flush()
                pbar.update(1)


def split_data(data, n_splits, MODEL_BASE):
    chunk_size = len(data) // n_splits
    chunks = [data[i*chunk_size : (i+1)*chunk_size] for i in range(n_splits)]
    chunks_with_model_name = []
    scale_size = n_splits // sum(MODEL_BASE.values())
    model_list = [key for key, count in MODEL_BASE.items() for _ in range(count * scale_size)]
    for i, chunk in enumerate(chunks):
        chunks_with_model_name.append((model_list[i], chunk))
    return chunks_with_model_name


if __name__ == "__main__":
    all_chunks = auto_read_data("/vepfs/wcf/G/zecheng/data/SlimPajama-6B/SlimPajama-6B.pkl")
    all_chunks = [(i, item) for i, item in enumerate(all_chunks)]
    
    # get process ids
    log_files = auto_read_dir("/vepfs/wcf/G/zecheng/data/SlimPajama-6B", file_suffix=".log")
    
    # filter generated samples
    processed_ids = [auto_read_data(log_file) for log_file in log_files]
    processed_ids = [int(i) for item in processed_ids for i in item]
    processed_ids = set(processed_ids)
    print_c(f"Already generatd samples {len(processed_ids)}\nBefore Filtering {len(all_chunks)}", "green")
    
    unprocessed_ids = set(range(len(all_chunks)))
    unprocessed_ids = list(unprocessed_ids - processed_ids)
    unprocessed_chunks = [all_chunks[i] for i in unprocessed_ids]
    print_c(f"After Filtering {len(unprocessed_chunks)}", "green")    
    
    all_chunks = unprocessed_chunks[:20000]
    
    MODEL_BASE = {"qwen-plus": 4, "qwen-max": 4, "qwen-max-0403": 1, "qwen-max-0107": 1}

    num_threads = 10
    chunks_with_model_name = split_data(all_chunks, num_threads, MODEL_BASE)
    threads = []
    
    for i in range(num_threads):
        chunk_length = len(chunks_with_model_name[i][1])
        thread = threading.Thread(target=process_chunk, args=(chunks_with_model_name[i], chunk_length, i))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()