from modelzipper.tutils import *
import re
from pprint import pprint
import itertools
import random
import transformers
from datasets import Dataset, concatenate_datasets, DatasetDict, load_from_disk, load_dataset
import pandas as pd
import numpy as np
import fire
import torch


def extract_qa_pairs(text):
    sp_text = text.split("####")
    if len(sp_text) < 9:
        return None
    q1, a1, q2, a2 = sp_text[2].strip(), sp_text[4].strip(), sp_text[6].strip(), sp_text[8].strip()
    # pattern = r"####Question \d+####(.*?)####Answer \d+####(.*?)(?=####Question \d+####|$)"
    # matches = re.findall(pattern, text, re.DOTALL)
   
    res = [{"question": q1, "answer": a1}, {"question": q2, "answer": a2}]
    # for i, match in enumerate(matches, 1):
    #     question_text, answer_text = match
    #     res.append({"question": question_text, "answer": answer_text})
    return res


def combine_data(data, chunk_num):
    combined_data = []
    for i in range(0, len(data), chunk_num):
        combined_data.append(data[i:i+chunk_num])
    return combined_data


def random_select_from_combined_data(all_samples, num_cases=8, selected_cases=1):
    SEP_TOKEN = ' [Doc] '
    # TEMPLATE = "{reference}\n\nQuestions: {question}"
    cases = list(range(num_cases))
    combinations_list = list(itertools.combinations(cases, selected_cases))

    batch_data = []
    ref_lst = [item['reference'] for item in all_samples]
    
    for item in combinations_list:
        chosen_id = item[0]
        remain_case_ids = list(set(cases) - set((chosen_id,)))
        reject_id = random.choice(remain_case_ids)
        for i in range(len(all_samples[chosen_id])):
            for j in range(len(all_samples[reject_id])):
                question = all_samples[chosen_id]['qa_pairs'][i]['question']
                cur_sample = {
                    "reference_list": ref_lst, 
                    "question": question,
                    # "prompt": TEMPLATE.format(reference=references, question=question), 
                    "chosen": all_samples[chosen_id]['qa_pairs'][i]['answer'], 
                    "rejected": all_samples[reject_id]['qa_pairs'][j]['answer'],
                    "chosen_span_id": chosen_id, 
                    "rejected_span_id": reject_id,
                }
                batch_data.append(cur_sample)

    return batch_data


def create_position_ids(N, L):
    """sampling N points from L (max_chunk_size space)"""
    if N == L:
        start_pos = 0
    else:
        start_pos = np.random.randint(0, L - N)
    end_pos = start_pos + N
    position_ids = torch.arange(start_pos, end_pos)
    return position_ids


def auto_padding(t: torch.Tensor, length: int, filling_value=-100, return_attention_mask=False):
    if length < t.size(0):
        if return_attention_mask:
            return t[:length]
        else:
            return t[:length], torch.ones_like(t[:length])

    padded_tensor = torch.full((length,), filling_value, dtype=t.dtype)
    padded_tensor[:t.size(0)] = t
    if return_attention_mask:
        attention_mask = torch.zeros(length, dtype=torch.int)
        attention_mask[:t.size(0)] = 1
        return padded_tensor, attention_mask
    return padded_tensor


def filter_fn_train(feature):
    lst = (np.array(feature['concatenated_labels'])!= -100).sum(-1).tolist()
    return all(n > 15 for n in lst)


def filter_fn_valid(feature):
    lst = (np.array(feature['concatenated_labels'])!= -100).sum(-1).tolist()
    return all(n > 0 for n in lst)


def filter_fn_input_length(feature):
        ipt_length = np.array(feature["concatenated_input_ids"]).shape[-1] 
        return ipt_length == 17408


def create_random_position_ipt_data(feature, max_chunk_size = 256, max_position_embeddings = 4096 * 16, max_qa_size = 1024):
    cut_refer_tok_lst = []
    for item in feature["refer_tok_lst"]:
        if len(item['input_ids'][0]) > max_chunk_size: 
            item['input_ids'][0] = item['input_ids'][0][:-(len(item['input_ids'][0]) - max_chunk_size)]
            item['attention_mask'][0] = item['attention_mask'][0][:-(len(item['input_ids'][0]) - max_chunk_size)]
        cut_refer_tok_lst.append(item)
    feature["refer_tok_lst"] = cut_refer_tok_lst

    # create reference index
    position_input_ids = []
    positional_chunks = list(range(0, max_position_embeddings - max_qa_size, 
        (max_position_embeddings - max_qa_size) // len(feature["refer_tok_lst"])))
    
    for i, item in enumerate(feature["refer_tok_lst"]):
        N = len(item['input_ids'][0])  # allocate for each instance
        chunk_ids = create_position_ids(N, max_chunk_size)
        chunk_ids += positional_chunks[i]
        position_input_ids.append(chunk_ids)

    concatenated_batch = {}
    reference_ids = [item["input_ids"][0] for item in feature["refer_tok_lst"]]
    reference_ids = [torch.tensor(item) for item in reference_ids]
    padded_input_attention_ids = [auto_padding(item, max_chunk_size, filling_value=0, return_attention_mask=True) for item in reference_ids]
    padded_reference_ids = [item[0] for item in padded_input_attention_ids]
    padded_reference_ids = torch.concatenate(padded_reference_ids, dim=0).unsqueeze(0)  # batch size = 1 as default
    padded_reference_attention_mask = [item[1] for item in padded_input_attention_ids]
    padded_reference_attention_mask = torch.concatenate(padded_reference_attention_mask, dim=0).unsqueeze(0)  # batch size = 1 as default
    padded_position_ids = [auto_padding(item, max_chunk_size, filling_value=0) for item in position_input_ids]
    padded_position_ids = torch.cat(padded_position_ids, dim=0).unsqueeze(0)

    # create question and input ids
    question_ids = torch.tensor(feature["question_tok"]["input_ids"])[0][1:]
    chosen_ans_ids = torch.tensor(feature["chosen_tok_ans"]["input_ids"])[0][1:max_qa_size - question_ids.size(-1)]  # cut the answer length
    rejected_ans_ids = torch.tensor(feature["reject_tok_ans"]["input_ids"])[0][1:max_qa_size - question_ids.size(-1)]  # cut the answer length

    question_chosen_ans_attention_ids = auto_padding(
        torch.concatenate([question_ids, chosen_ans_ids], dim=0), 
        max_qa_size, filling_value=0, return_attention_mask=True
    )
    question_rejected_ans_attention_ids = auto_padding(torch.concatenate(
        [question_ids, rejected_ans_ids], dim=0), 
        max_qa_size, filling_value=0, return_attention_mask=True
    )

    question_chosen_ans_ids = question_chosen_ans_attention_ids[0].unsqueeze(0)
    question_chosen_ans_attention_mask = question_chosen_ans_attention_ids[1].unsqueeze(0)
    question_chosen_ans_position_ids = torch.arange(
        0, question_chosen_ans_attention_mask.sum()) + (max_position_embeddings - max_qa_size)  # bias for positional ids
    question_chosen_ans_position_ids = auto_padding(
        question_chosen_ans_position_ids, length=max_qa_size, filling_value=0)

    question_rejected_ans_ids = question_rejected_ans_attention_ids[0].unsqueeze(0)
    question_rejected_ans_attention_mask = question_rejected_ans_attention_ids[1].unsqueeze(0)
    question_rejected_ans_position_ids = torch.arange(
        0, question_rejected_ans_attention_mask.sum()) + (max_position_embeddings - max_qa_size)  # bias for positional ids
    question_rejected_ans_position_ids = auto_padding(
        question_rejected_ans_position_ids, length=max_qa_size, filling_value=0)

    # create labels
    reference_labels = torch.full_like(padded_reference_attention_mask, -100)
    question_chosen_labels = torch.full_like(question_chosen_ans_position_ids, -100)
    question_chosen_labels[question_ids.size(-1): question_ids.size(-1) + chosen_ans_ids.size(-1)] = chosen_ans_ids
    question_chosen_labels = question_chosen_labels.unsqueeze(0)
    
    question_rejected_labels = torch.full_like(question_rejected_ans_position_ids, -100)
    question_rejected_labels[question_ids.size(-1): question_ids.size(-1) + rejected_ans_ids.size(-1)] = rejected_ans_ids
    question_rejected_labels = question_rejected_labels.unsqueeze(0)
    
    # wrap in the dict
    chosen_input_ids = torch.concat([padded_reference_ids, question_chosen_ans_ids], dim=1)
    rejected_input_ids = torch.concat([padded_reference_ids, question_rejected_ans_ids], dim=1)
    concatenated_batch["concatenated_input_ids"] = torch.concat([chosen_input_ids, rejected_input_ids], dim=0)
    
    chosen_attention_mask = torch.concat(
        [padded_reference_attention_mask, question_chosen_ans_attention_mask], dim=1)
    rejected_attention_mask = torch.concat(
        [padded_reference_attention_mask, question_rejected_ans_attention_mask], dim=1)
    concatenated_batch["concatenated_attention_mask"] = torch.concat(
        [chosen_attention_mask, rejected_attention_mask], dim=0)

    chosen_labels = torch.concat([reference_labels, question_chosen_labels], dim=1)
    rejected_labels = torch.concat([reference_labels, question_rejected_labels], dim=1)
    concatenated_batch["concatenated_labels"] = torch.concat([chosen_labels, rejected_labels], dim=0)

    chosen_position_ids = torch.concat([padded_position_ids, question_chosen_ans_position_ids.unsqueeze(0)], dim=1)
    rejected_position_ids = torch.concat([padded_position_ids, question_rejected_ans_position_ids.unsqueeze(0)], dim=1)
    concatenated_batch["position_ids"] = torch.concat([chosen_position_ids, rejected_position_ids], dim=0)
    return concatenated_batch


def stage_0(all_data, save_path, tokenizer, chunk_num, num_proc):
    processed_data = []
    for item in all_data:
        ref, qa_pairs = item['reference'], item['qa_pairs']
        processed_data.append({"reference": ref, "qa_pairs": qa_pairs})

    combined_data = combine_data(processed_data, chunk_num=chunk_num)
    print(len(combined_data))
    all_created_cases = []

    with tqdm(total=len(combined_data)) as pbar:
        for c_data in combined_data:
            batch_data = random_select_from_combined_data(c_data, num_cases=len(c_data), selected_cases=1)
            all_created_cases += batch_data
            pbar.update(1)

    print(f"finish, current data sample nums: {len(all_created_cases)}")
    print(all_created_cases[0].keys())

    train_dataset, valid_dataset = all_created_cases[500:], all_created_cases[:500]
    train_df = pd.DataFrame(train_dataset)
    valid_df = pd.DataFrame(valid_dataset)
    trans_train_datasets = Dataset.from_pandas(train_df, split="train")
    trans_valid_datasets = Dataset.from_pandas(valid_df, split="valid")
    
    
    def tokenize_row(feature):
        reference_lst, question = feature['reference_list'], feature['question']
        chosen_ans, rejected_ans = feature["chosen"], feature["rejected"]
        chosen_span_id, rejected_span_id = feature['chosen_span_id'], feature['rejected_span_id']

        refer_tok_lst, question_tok = [tokenizer(ref, return_tensors="pt") for ref in reference_lst], tokenizer(question, return_tensors="pt")
        chosen_tok_ans, reject_tok_ans = tokenizer(chosen_ans, return_tensors="pt"), tokenizer(rejected_ans, return_tensors="pt")

        return {
            "refer_tok_lst": refer_tok_lst, "question_tok": question_tok,
            "chosen_tok_ans": chosen_tok_ans, "reject_tok_ans": reject_tok_ans,
            "chosen_span_id": chosen_span_id, "rejected_span_id": rejected_span_id
        }

    print("begin tokenizing row")
    trans_train_datasets = trans_train_datasets.map(tokenize_row, num_proc=num_proc)
    trans_valid_datasets = trans_valid_datasets.map(tokenize_row, num_proc=num_proc)
    combined_datasets = DatasetDict({"train": trans_train_datasets, "valid": trans_valid_datasets})
    auto_mkdir(save_path) 
    print(f"save combined datasets to {save_path}")
    combined_datasets.save_to_disk(save_path)
    print("dataset structure:")
    print(combined_datasets['valid'])


def stage_1(all_data, save_path, num_proc):
    print("execute stage 1")
    print(all_data)

    all_data = all_data.map(
        create_random_position_ipt_data, num_proc=64, 
        remove_columns=['reference_list', 'question', 'chosen', 'rejected', 'refer_tok_lst', 'question_tok', 'chosen_tok_ans', 'reject_tok_ans']
    )
    filter_train_data = all_data['train'].filter(filter_fn_train, num_proc=num_proc)
    filter_valid_data = all_data['valid'].filter(filter_fn_valid, num_proc=num_proc)
    filter_data = DatasetDict({"train": filter_train_data, "valid": filter_valid_data})

    print("before filtering")
    print(all_data)
    print("after filtering")
    print(filter_data)

    filter_data = all_data.filter(filter_fn_input_length, num_proc=num_proc)
    print(filter_data)
    filter_data.save_to_disk(save_path)




def main(raw_file_path: None, save_dir: None, stage: int=0, chunk_num: int=64, model_name_or_path: str="/data/zecheng/hf_models/Llama-2-7b-hf"):
    """
    preprocess file from raw data to hf dataset
    0. load and save raw data
    1. process datasets
    """
    if raw_file_path is None: raw_file_path = "/data/zecheng/data/SlimPajama-6B/filtered_wo_ana"
    if save_dir is None: save_dir = "/data/zecheng/data/SlimPajama-6B"

    save_path = os.path.join(save_dir, f"dpo_data/hf_data_{chunk_num}_stage{stage}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    if stage == 0: 
        print("stage 0: load and save raw data")
        all_data = load_from_disk(raw_file_path)
        stage_0(all_data, save_path, tokenizer, chunk_num=chunk_num, num_proc=64)
    
    elif stage == 1: 
        print("stage 1: process datasets")
        load_path = os.path.join(save_dir, f"dpo_data/hf_data_{chunk_num}_stage{stage-1}")
        all_data = load_from_disk(load_path)
        print(f"total pairs: {len(all_data)}")
        stage_1(all_data, save_path, num_proc=64)


if __name__ == "__main__":
    fire.Fire(main)