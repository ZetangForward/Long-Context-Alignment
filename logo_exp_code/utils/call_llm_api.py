from volcenginesdkarkruntime import Ark
from modelzipper.tutils import *
import fire
import datasets
import transformers
import collections
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

api_client = Ark(api_key="ea28bf46-979c-49b9-b08a-92303bb99052")

MODEL_ENDPOINT = {
    "doubao-lite-4k": "ep-20240618124048-xd5vm",
    "doubao-pro-4k": "ep-20240618125023-lkmzs",
    "doubao-pro-32k": "ep-20240618163715-nmkbp",
    "doubao-pro-128k": "ep-20240822215215-46jsv",   
}

SYSTEM_MESSAGE = """You will be provided with a text snippet (reference) and asked to generate one question and answer pair based on the reference. Each pair should consist of a question, an answer, and a reference snippet localized to the raw reference.

    Determine if the input content is in English. If the content contains a lot of code, gibberish, math symbols, or HTTP URLs and is not suitable for generating questions and answers, respond with the word: Null.

    If the content is suitable for generating questions and answers, return the output in the following format:

    ####Q: {Question}####A: {Answer}####R: {Reference}####

    {Question}: A question generated from the reference.
    {Answer}: The corresponding answer, which can be inferred from the content of the reference.
    {Reference}: A short, exact excerpt from the original text that directly relates to the question and answer. This snippet should not be identical to the answer but should support the answer. Ensure the reference snippet is concise and as short as possible, directly taken from the original text, and not the same as the answer.

    You must follow the format provided above.
    """

def call_with_messages(client, model_id, user_query, max_attempts=2, max_new_tokens=2048):
    attempts = 0
    while attempts < max_attempts:
        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": user_query},
                ],
                max_tokens=max_new_tokens,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Attempt {attempts + 1} failed: {e}")
            attempts += 1
            if attempts < max_attempts:
                time.sleep(5)  # sleep for 5 seconds before retrying
            else:
                print("Failed after maximum attempts.")
                return None
            
def load_raw_data(file_path, tokenizer, num_process=12, chunk_size = 2048):
    print("load raw dataset")
    raw_data = datasets.load_dataset(file_path, num_proc=num_process)
    print(raw_data)

    # step 1: merge and split dataset
    print("merge and split the datasets")
    def combine_splits(dataset_dict):
        combined_texts = []
        for split in ['train', 'validation', 'test']:
            combined_texts.extend(dataset_dict[split]['text'])
        return {"text": combined_texts}
    combined_data = datasets.DatasetDict({
        "combined": datasets.Dataset.from_dict(combine_splits(raw_data))
    })

    # step 2: split_into_chunks
    print("split the dataset into chunks")
    def split_into_chunks(example):
        token_ids = tokenizer(example['text'], return_tensors="pt")["input_ids"][0]
        chunks = []
        for i in range(0, token_ids.size(-1), chunk_size):
            chunk = token_ids[i:i + chunk_size]
            if chunk.size(-1) == chunk_size:
                chunks.append(tokenizer.decode(chunk))
        return {"chunks": chunks, "num_chunks": len(chunks)}
    chunked_data = combined_data["combined"].map(split_into_chunks, remove_columns=["text"])
    
    # step 3: filter short data
    def filter_short_data(batch): 
        return len(batch["chunks"]) >= 2
    flattened_data = chunked_data.filter(filter_short_data)
    return flattened_data


def flatten_raw_data(data):
    flatten_data = []
    group_pairs = {}
    i = 0
    j = 0
    for sample in data:
        group_pairs[j] = []
        for chunk in sample["chunks"]:
            flatten_data.append((chunk, i))
            i += 1
            group_pairs[j].append(i)
        j += 1
    return flatten_data, group_pairs


"""multi-threading processing"""

# def process_sample(sample, client, model_name, save_dir):
#     chunk, idx = sample
#     model_gen = call_with_messages(client, MODEL_ENDPOINT[model_name], chunk)
#     # Generate a unique filename for each thread
#     thread_name = threading.current_thread().name
#     save_path = os.path.join(save_dir, f"{thread_name}_doubao_gen.jsonl")
    
#     # Use thread-specific progress bar
#     with tqdm(total=1, desc=f"Thread {thread_name}") as pbar:
#         with open(save_path, "a") as f:
#             f.write(json.dumps({"id": idx, "model_gen": model_gen}) + "\n")
#         pbar.update(1)

def process_chunk(chunk, chunk_length, thread_id, model_id):
    output_file = f"/data/zecheng/data/process_wiki_document/one_hop/generated_QA_pairs_thread_{thread_id}.jsonl"
    log_file = f"/data/zecheng/data/process_wiki_document/one_hop/logs/processing_log_thread_{thread_id}.log"
    with open(output_file, 'a') as f, open(log_file, 'a') as log_f:
        with tqdm(total=chunk_length, desc=f"Processing thread {thread_id}") as pbar:
            for sample in chunk:
                response = call_with_messages(client=api_client, model_id=model_id, user_query=sample[0])
                if response is None:
                    continue
                else:  # has response
                    pairs = {"id": sample[1], "reference": sample[0], "QA_pairs": response}
                    json_str = json.dumps(pairs)
                    # Write to output file
                    f.write(json_str + '\n')
                    f.flush()
                    # Write to log file
                    log_f.write(f"{sample[1]}\n")
                    log_f.flush()
                pbar.update(1)



def split_data(data, n_splits):
    # data = data.to_dict()
    # data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
    chunk_size = len(data) // n_splits
    chunks = [data[i*chunk_size : (i+1)*chunk_size] for i in range(n_splits)]
    return chunks


def call_doubao_api(model_id, query, max_attempts=2):
    '''
    1. first choose the model_id: doubao-pro-4k
    2. enter the query
    3. max_attempts, if exceed max_attempts, return None
    '''
    return call_with_messages(client, model_id, query, max_attempts)
    

def main(
        file_path: str = "/data/zecheng/data/wikitext_document_level-103_reformat", 
        save_dir: str = "/data/zecheng/data/process_wiki_document", 
        model_name: str = "doubao-pro-32k",
        num_thread: int = 12, 
        model_name_or_path: str = "/data/zecheng/hf_models/Llama-2-7b-hf",
        chunk_size: int = 4096,
        chunk_data_path: str = "/data/zecheng/data/process_wiki_document",
        hop_type: str = "one-hop",
    ):

    # init client
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    
    # load and process raw dataset
    if not os.path.exists(chunk_data_path):
        print("load and process data into chunks")
        raw_data = load_raw_data(file_path, tokenizer, num_process=num_thread, chunk_size=chunk_size)
        raw_data.save_to_disk(chunk_data_path)
    else:
        print("load chunk datasets directly")
        raw_data = datasets.load_from_disk(chunk_data_path)

    num_chunks_distribution = collections.Counter(raw_data['num_chunks'])
    for num_chunks, count in num_chunks_distribution.items():
        print(f"Number of chunks: {num_chunks}, Count: {count}")
    
    if hop_type == "one-hop":
        if 0:
            flatten_data = flatten_raw_data(raw_data)
            auto_mkdir(os.path.join(save_dir, "one_hop"))
            save_path = os.path.join(f"{save_dir}/one_hop", f"doubao_gen.jsonl")
            with open(save_path, "a") as f, tqdm(total=len(flatten_data)) as pbar:
                for sample in flatten_data:
                    chunk, idx = sample
                    model_gen = call_with_messages(api_client, MODEL_ENDPOINT[model_name], chunk)
                    f.write(json.dumps({"id": idx, "model_gen": model_gen}) + "\n")
                    pbar.update(1)
        else: # multi-threading
            num_threads = 16
            auto_mkdir(os.path.join(save_dir, "one_hop"))
            save_dir = os.path.join(save_dir, "one_hop")  # Update save_dir
            flatten_data, group_info = flatten_raw_data(raw_data)
            data_splits = split_data(flatten_data, num_threads)
            threads = []

            for i in range(num_threads):
                chunk_length = len(data_splits[i])
                thread = threading.Thread(target=process_chunk, args=(data_splits[i], chunk_length, i, MODEL_ENDPOINT[model_name], api_client))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()

if __name__ == "__main__":
    fire.Fire(main)