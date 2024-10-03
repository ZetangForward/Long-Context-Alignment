from modelzipper.tutils import *
from utils import *
from fire import Fire
import sys
sys.path.append("/data/zecheng/sunzc/FlagEmbedding/Long_LLM/longllm_qlora")
from src import apply_chat_template

def data_process(tokenizer, sample, max_seq_length, device):
    TEMPLATE = """There is an important infomation hidden in the following context. Find the information and memorize it. I will quiz you about the important information there.\n{evidence}\n{question}\n"""
    
    question, answer = sample['question'], sample['answer']
    placeholder_length = 256  # for question and answers
    
    all_evidences = [" [DOC] ".join(item[0]).strip() for item in sample["all_evidence"]]
    evidence_str = " [DOC] ".join(all_evidences)

    input_str = TEMPLATE.format(evidence=evidence_str, question=question)
    
    tokenized_evidence = tokenizer(input_str, return_tensors="pt").input_ids
    context_sequence_length = max_seq_length - placeholder_length

    if tokenized_evidence.size(-1) > context_sequence_length:
        exceed_length = tokenized_evidence.size(-1) - context_sequence_length
        cut_lengths = [int((tokenizer(s, return_tensors="pt", add_special_tokens=False).input_ids.size(-1) / tokenized_evidence.size(-1)) * exceed_length) for s in all_evidences]

        sum_cut_lengths = sum(cut_lengths)
        if sum_cut_lengths != exceed_length:
            cut_lengths[-1] += (exceed_length - sum_cut_lengths)

        tokenized_evidences = []
        for s, cut_length in zip(all_evidences, cut_lengths):
            tok_s = tokenizer(s, return_tensors="pt", add_special_tokens=False).input_ids[0, :-cut_length]
            tokenized_evidences.append(tokenizer.decode(tok_s, skip_special_tokens=True))
        
        evidence_str = " [DOC] ".join(tokenized_evidences)
        input_str = TEMPLATE.format(evidence=evidence_str, question=question)
        prompt = apply_chat_template(
                        "llama-3", 
                        messages=[{'role': 'user', 'content': input_str}],
                        tokenizer=tokenizer,
                        add_generation_prompt=True,
                    ).raw

        tokenized_evidence = tokenizer(prompt, return_tensors="pt").to(device).input_ids

    return tokenized_evidence, answer


def main(model_path: str = None, peft_path: str  = None, save_dir: str = None, rope_theta: int = None, rope_factor: int = None, rope_type: int = None, model_name: str = None, max_position_embeddings: int = 65536, testing_data_path: str = "./fanoutqa_data/open_book_testing_sets.jsonl"):

    if save_dir is None:
        save_dir = "./fanoutqa/results"
    auto_mkdir(save_dir)
    
    testing_data = auto_read_data(testing_data_path)
    
    if model_name is None: 
        model_name = os.path.basename(model_path)
    
    # load models
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_model = load_model(model_path, peft_path, rope_theta, rope_factor, rope_type, max_position_embeddings, max_training_length=16384)
    
    save_file_path = os.path.join(save_dir, f"{model_name}.jsonl")
    
    f_context = None

    with tqdm(total=len(testing_data)) as pbar, open(save_file_path, "w") as f:
        for sample in testing_data:
            input_ids, answer = data_process(tokenizer, sample, max_seq_length=60000, device=test_model.device)
            outputs = test_model.generate(
                input_ids.to(test_model.device), 
                num_beams=1,
                min_new_tokens=1,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                use_cache=True, 
                max_new_tokens=128
            )
            pred_str = tokenizer.decode(outputs[0][input_ids.shape[-1]:])
            saved_content = json.dumps({"pred_str": pred_str, "answer": answer}) + '\n'
            
            if f_context is not None:
                context_str = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                f_context.write(json.dumps({"context": context_str}) + '\n')
                f_context.flush()

            pbar.update(1)
            f.write(saved_content)
            f.flush()
    
    f_context.close()

if __name__ == "__main__":

    Fire(main)
    





