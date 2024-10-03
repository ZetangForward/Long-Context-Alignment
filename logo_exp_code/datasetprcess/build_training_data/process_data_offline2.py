import os
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional
from datasets import Dataset, load_dataset, load_from_disk, DatasetDict
from peft import LoraConfig, PeftModel, get_peft_model
import transformers
from accelerate import Accelerator
from contextlib import contextmanager, nullcontext
import torch.nn as nn
from typing import List, Tuple, Union, Literal, Dict
from modelzipper.tutils import *
import torch.nn.functional as F
import torch.nn as nn
from scipy.stats import qmc
import numpy as np
import itertools
from functools import partial
from accelerate import PartialState
import pandas as pd
import concurrent.futures

def find_index(id_lst, prefix_id, suffix_id):
    return id_lst.index(prefix_id), id_lst.index(suffix_id)


def create_position_ids(N, L):
    """sampling N points from L (max_chunk_size space)"""
    if N == L:
        start_pos = 0
    else:
        start_pos = np.random.randint(0, L - N)
    end_pos = start_pos + N
    position_ids = torch.arange(start_pos, end_pos)
    return position_ids


def create_covering_position_ids(N, L):
    """Create sets of position IDs to cover all positions from 0 to L-1 with intervals of length N."""
    if N > L:
        raise ValueError("N should not be greater than L")
    num_intervals = (L + N - 1) // N
    position_ids_list = []
    for i in range(num_intervals):
        start_pos = i * (L - N) // (num_intervals - 1) if num_intervals > 1 else 0
        end_pos = start_pos + N
        if end_pos > L:
            end_pos = L
            start_pos = L - N if L > N else 0
        position_ids = torch.arange(start_pos, end_pos)
        position_ids_list.append(position_ids)
    return position_ids_list

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


def combine_fn(lst, max_candidates=2, max_combination=16):
    trimmed_lists = [random.sample(sublst, min(len(sublst), max_candidates)) if len(sublst) > max_candidates else sublst for sublst in lst]
    all_combinations = itertools.product(*trimmed_lists)
    concatenated_results = [torch.cat(combination) for combination in all_combinations]
    concatenated_results = random.sample(concatenated_results, min(len(concatenated_results), max_combination))
    return concatenated_results

def create_system_suffix(tokenizer, system_suffix, special_token_id: int=13):
    tok_suffix = tokenizer(system_suffix, return_tensors="pt", add_special_tokens=False)
    padded_tok_suffix, padded_suffix_attention_mask = tok_suffix.input_ids[0], tok_suffix.attention_mask[0]
    # add special token
    padded_tok_suffix = torch.concatenate([padded_tok_suffix, torch.tensor([special_token_id])], dim=0)
    padded_suffix_attention_mask = torch.concatenate([padded_suffix_attention_mask, torch.tensor([1])], dim=0)
    system_position_ids = torch.arange(0, padded_tok_suffix.size(-1))
    return padded_tok_suffix, padded_suffix_attention_mask, system_position_ids

def create_chunked_reference(tokenizer: AutoTokenizer, all_refs: List[str], real_reference_size: int, max_embedding_size: int, system_prompt_size: int, qa_size: int, special_token_id: int=13):
    real_max_chunk_size = real_reference_size // len(all_refs) - 1 # allocate one position for attention reallocation
    tok_all_ref = [tokenizer(item, return_tensors="pt", add_special_tokens=False).input_ids[0] for item in all_refs]
    truncted_refer_tok_lst, statistic_data_size = [], []
    for item in tok_all_ref:
        statistic_data_size.append(item.size(-1))
        if item.size(-1) > real_max_chunk_size: 
            item = item[: real_max_chunk_size]
        truncted_refer_tok_lst.append(item)

    fake_position_chunk_size = (max_embedding_size - qa_size - system_prompt_size) // len(tok_all_ref)  # with last special token index for each chunk
    positional_chunks = torch.arange(system_prompt_size, max_embedding_size - qa_size, fake_position_chunk_size)
    # Here, end_positional_chunks denotes special token ids
    begin_positional_chunks, end_positional_chunks = positional_chunks[:-1], positional_chunks[1:] - 1  
    all_chunk_pos_lst = []
    
    for i, item in enumerate(truncted_refer_tok_lst):
        chunk_token_pos_lst = create_covering_position_ids(item.size(-1), fake_position_chunk_size-1)
        chunk_token_pos_lst = [item + begin_positional_chunks[i] for item in chunk_token_pos_lst]
        all_chunk_pos_lst.append(chunk_token_pos_lst)

    padded_chunk_pos_lst = [[auto_padding(sub_item, real_max_chunk_size, filling_value=0, return_attention_mask=False) for sub_item in item] for item in all_chunk_pos_lst]
    padded_refer_tok_lst = [auto_padding(item, real_max_chunk_size, filling_value=0, return_attention_mask=True) for item in truncted_refer_tok_lst]
    padded_refer_tok_ids = [item[0] for item in padded_refer_tok_lst]
    padded_refer_attention_mask = [item[1] for item in padded_refer_tok_lst]

    candicated_padded_position_ids = []
    padded_ref_input_ids_lst, padded_ref_attention_mask_lst = [], []
    
    for chunk_pos_ids, chunk_spe_pos_lst in zip(end_positional_chunks, padded_chunk_pos_lst):
        tmp_chunk_pos_ids = []
        for tmp in chunk_spe_pos_lst:
            tmp = torch.concatenate([tmp, torch.tensor([chunk_pos_ids])], dim=0)
            tmp_chunk_pos_ids.append(tmp)  # [[0,1,...,C1], [C2,C2+1,...,C3], ...]
        candicated_padded_position_ids.append(tmp_chunk_pos_ids)
    candicated_padded_position_ids = combine_fn(candicated_padded_position_ids, max_combination=32)

    for padded_chunk_tok_ref_input_ids, padded_chunk_tok_ref_attention_mask in zip(padded_refer_tok_ids, padded_refer_attention_mask):
        padded_chunk_tok_ref_input_ids = torch.concatenate([padded_chunk_tok_ref_input_ids, torch.tensor([special_token_id])], dim=0)
        padded_chunk_tok_ref_attention_mask = torch.concatenate([padded_chunk_tok_ref_attention_mask, torch.tensor([1])], dim=0)
        padded_ref_input_ids_lst.append(padded_chunk_tok_ref_input_ids)
        padded_ref_attention_mask_lst.append(padded_chunk_tok_ref_attention_mask)
    
    padded_ref_input_ids = torch.concatenate(padded_ref_input_ids_lst, dim=0)
    padded_ref_attention_mask = torch.concatenate(padded_ref_attention_mask_lst, dim=0)
    all_spe_pos = torch.arange(real_max_chunk_size, real_reference_size, real_max_chunk_size + 1)

    return candicated_padded_position_ids, padded_ref_input_ids, padded_ref_attention_mask, all_spe_pos


def create_qa(QUESTION_TEMPLATE, ANSWER_TEMPLATE, combined_question, combined_answer, prefix_a: str, suffix_a: str, last_position: int, qa_size: int, special_token_id: int):
    """
    last_position 是reference position ids最大的数值，下面的代码要加一个 last_position + 1的shift
    qa_size 规定了最大的qa 的长度，所以总长度需要手动卡一下
    """
    # Create Question
    question = QUESTION_TEMPLATE.format(question=combined_question)
    tok_question = tokenizer(question, return_tensors="pt", add_special_tokens=False).input_ids[0]
    padded_tok_question, padded_question_attention_mask = auto_padding(tok_question, tok_question.size(-1), filling_value=0, return_attention_mask=True)
    padded_tok_question = torch.concatenate([padded_tok_question, torch.tensor([special_token_id])], dim=0)
    padded_question_attention_mask = torch.concatenate([padded_question_attention_mask, torch.tensor([1])], dim=0)
    question_position_input_ids = create_position_ids(tok_question.size(-1), tok_question.size(-1)) + last_position + 1
    last_pos = question_position_input_ids.max() + 1
    question_position_input_ids = torch.concatenate([question_position_input_ids, torch.tensor([last_pos])], dim=0)
    spe_tok_pos = question_position_input_ids.size(-1) - 1

    # Create Chosen / Rejected Answers / and their labels
    chosen_answer = ANSWER_TEMPLATE.format(answer=combined_answer)
    prefix_rejected_answer = ANSWER_TEMPLATE.format(answer=prefix_a)
    suffix_rejected_answer = ANSWER_TEMPLATE.format(answer=suffix_a)
    tok_chosen_answer = tokenizer(chosen_answer, return_tensors="pt", add_special_tokens=False).input_ids[0]
    tok_prefix_rejected_answer = tokenizer(prefix_rejected_answer, return_tensors="pt", add_special_tokens=False).input_ids[0]
    tok_suffix_rejected_answer = tokenizer(suffix_rejected_answer, return_tensors="pt", add_special_tokens=False).input_ids[0]

    system_reference_question_size = last_position + 1 + question_position_input_ids.size(-1)

    padded_tok_chosen_answer, padded_chosen_answer_attention_mask = auto_padding(tok_chosen_answer, qa_size-padded_question_attention_mask.size(-1), filling_value=0, return_attention_mask=True)
    tok_chosen_answer_labels = auto_padding(tok_chosen_answer, qa_size - padded_question_attention_mask.size(-1), filling_value=-100)
    chosen_answer_position_ids = create_position_ids(tok_chosen_answer.size(-1), tok_chosen_answer.size(-1)) + system_reference_question_size
    chosen_answer_position_ids = auto_padding(chosen_answer_position_ids, qa_size - padded_question_attention_mask.size(-1), filling_value=0)

    padded_tok_prefix_rejected_answer, padded_prefix_rejected_answer_attention_mask = auto_padding(tok_prefix_rejected_answer, qa_size - padded_question_attention_mask.size(-1), filling_value=0, return_attention_mask=True)
    tok_prefix_rejected_answer_labels = auto_padding(tok_prefix_rejected_answer, qa_size - padded_question_attention_mask.size(-1), filling_value=-100)
    prefix_rejected_answer_position_ids = create_position_ids(tok_prefix_rejected_answer.size(-1), tok_prefix_rejected_answer.size(-1)) + system_reference_question_size
    prefix_rejected_answer_position_ids = auto_padding(prefix_rejected_answer_position_ids, qa_size - padded_question_attention_mask.size(-1), filling_value=0)
    
    padded_tok_suffix_rejected_answer, padded_suffix_rejected_answer_attention_mask = auto_padding(tok_suffix_rejected_answer, qa_size - padded_question_attention_mask.size(-1), filling_value=0, return_attention_mask=True)
    tok_suffix_rejected_answer_labels = auto_padding(tok_suffix_rejected_answer, qa_size - padded_question_attention_mask.size(-1), filling_value=-100)
    suffix_rejected_answer_position_ids = create_position_ids(tok_suffix_rejected_answer.size(-1), tok_suffix_rejected_answer.size(-1)) + system_reference_question_size
    suffix_rejected_answer_position_ids = auto_padding(suffix_rejected_answer_position_ids, qa_size - padded_question_attention_mask.size(-1), filling_value=0)
 
    return padded_tok_question, padded_question_attention_mask, question_position_input_ids, \
        padded_tok_chosen_answer, padded_chosen_answer_attention_mask, tok_chosen_answer_labels, chosen_answer_position_ids, \
        padded_tok_prefix_rejected_answer, padded_prefix_rejected_answer_attention_mask, tok_prefix_rejected_answer_labels, prefix_rejected_answer_position_ids, \
        padded_tok_suffix_rejected_answer, padded_suffix_rejected_answer_attention_mask, tok_suffix_rejected_answer_labels, suffix_rejected_answer_position_ids, spe_tok_pos
    

def create_covering_position_ipt_data(tokenizer, all_refs: List[str], combined_question: str, combined_answer: str, prefix_a: str, suffix_a: str, qa_size: int, max_embedding_size: int, real_reference_size: int, special_token_id: int = None, prefix_id: int = None, suffix_id: int = None):
    statistic_data_size = []

    SYSTEM_SUFFIX = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBelow is some references. Please read it carefully and answer the following question.<|eot_id|>"
    QUESTION_TEMPLATE = "<|start_header_id|>user<|end_header_id|>\n\nPlease answer the following question according to the references: {question}<|eot_id|>"
    ANSWER_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>\n\nThe answer is: {answer}<|eot_id|><|end_of_text|>"

    # Create System Suffix
    padded_tok_input_ids_system_suffix, padded_attention_mask_system_suffix, padded_position_ids_system_suffix = create_system_suffix(tokenizer, SYSTEM_SUFFIX, special_token_id)
    system_prompt_size = padded_attention_mask_system_suffix.size(-1)
    all_spe_pos = [system_prompt_size-1]
    # create chunk reference (input_ids, attention_mask and positional ids)
    candicated_padded_position_ids_lst, padded_ref_input_ids, padded_ref_attention_mask, ref_spe_pos = create_chunked_reference(tokenizer, all_refs, real_reference_size, max_embedding_size, system_prompt_size, qa_size, special_token_id)

    ref_spe_pos += system_prompt_size
    all_spe_pos.extend(ref_spe_pos.tolist())

    # combine and wrap each position_id, input_ids and attention_mask
    last_position = max_embedding_size - qa_size  # size for real reference and system prompt

    # Create Question, all Answers
    padded_tok_question, padded_question_attention_mask, \
    question_position_input_ids, padded_tok_chosen_answer, \
    padded_chosen_answer_attention_mask, tok_chosen_answer_labels, \
    chosen_answer_position_ids, padded_tok_prefix_rejected_answer, \
    padded_prefix_rejected_answer_attention_mask, tok_prefix_rejected_answer_labels, \
    prefix_rejected_answer_position_ids, padded_tok_suffix_rejected_answer, \
    padded_suffix_rejected_answer_attention_mask, tok_suffix_rejected_answer_labels, \
    suffix_rejected_answer_position_ids, spe_tok_pos = create_qa(
        QUESTION_TEMPLATE, ANSWER_TEMPLATE, combined_question, combined_answer, prefix_a, suffix_a, last_position, qa_size, special_token_id=special_token_id
    )
    all_spe_pos.append(spe_tok_pos + system_prompt_size + padded_ref_input_ids.size(-1))
    all_datasets = []  # different combination of positions 

    for i, ref_position_id in enumerate(candicated_padded_position_ids_lst):
        concatenated_batch = {}
        concatenated_batch["input_ids"] = torch.concatenate([padded_tok_input_ids_system_suffix, padded_ref_input_ids, padded_tok_question], dim=0)
        concatenated_batch["attention_mask"] = torch.concatenate([padded_attention_mask_system_suffix, padded_ref_attention_mask, padded_question_attention_mask], dim=0)
        concatenated_batch["position_ids"] = torch.concatenate([padded_position_ids_system_suffix, ref_position_id, question_position_input_ids], dim=0)
        referece_question_length = concatenated_batch["attention_mask"].size(-1)
        concatenated_batch["all_spe_pos"] = all_spe_pos
        referece_question_labels = torch.full((1, referece_question_length), -100)[0]
        
        # Create Labels for Each Part
        concatenated_batch["chosen_answer"] = {
            "input_ids": padded_tok_chosen_answer, 
            "attention_mask": padded_chosen_answer_attention_mask, 
            "labels": torch.concatenate([referece_question_labels, tok_chosen_answer_labels], dim=0),
            "position_ids": chosen_answer_position_ids
        }
        concatenated_batch["prefix_rejected_answer"] = {
            "input_ids": padded_tok_prefix_rejected_answer, 
            "attention_mask": padded_prefix_rejected_answer_attention_mask, 
            "labels": torch.concatenate([referece_question_labels, tok_prefix_rejected_answer_labels], dim=0),
            "position_ids": prefix_rejected_answer_position_ids,
        }
        concatenated_batch["suffix_rejected_answer"] = {
            "input_ids": padded_tok_suffix_rejected_answer, 
            "attention_mask": padded_suffix_rejected_answer_attention_mask, 
            "labels": torch.concatenate([referece_question_labels, tok_suffix_rejected_answer_labels], dim=0),
            "position_ids": suffix_rejected_answer_position_ids,
        }
        if prefix_id is not None and suffix_id is not None:
            concatenated_batch["chosen_ids"] = (prefix_id, suffix_id)
        all_datasets.append(concatenated_batch)
        statistic_data_size.append(concatenated_batch["input_ids"].size(-1))

    return all_datasets, sum(statistic_data_size) / len(statistic_data_size)


def convert_jsonl_to_dict(data, format="dict"):
    """
    Convert a list of dictionaries to a Hugging Face datasets.Dataset object.
    
    Args:
        data (List[Dict]): The list of dictionaries to be converted.

    Returns:
        datasets.Dataset: The converted dataset.
    """
    convert_dict = {}
    for k in data[0].keys():
        convert_dict[k] = [item[k] for item in data]
    if format == "hf":
        convert_dict = datasets.Dataset.from_dict(convert_dict)
    return convert_dict

def process_item(item):
    return [sub_ for sub_ in item]

def process_data_parallel(processed_data, max_workers=4):
    final_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_item, item) for item in processed_data]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(processed_data)):
            final_data.extend(future.result())
    return final_data

def map_fn(item, tokenizer, special_token_id, qa_size, max_embedding_size, real_reference_size, prefix_id, suffix_id):
    # 暂时没有使用，只是为了大数据的快速构造而写的code
    all_ref_text = item["all_ref_text"]
    combined_question, final_answer = item["combined_question"], item["final_answer"]
    prefix_q, suffix_q = item["prefix_q"], item["suffix_q"]
    prefix_a, suffix_a = item["prefix_a"], item["suffix_a"]
    prefix_id, suffix_id = None, None
    all_datasets, ref_length = create_covering_position_ipt_data(tokenizer, all_ref_text, combined_question, final_answer, prefix_a, suffix_a, qa_size=qa_size, max_embedding_size=max_embedding_size, real_reference_size=real_reference_size, special_token_id=special_token_id, prefix_id=prefix_id, suffix_id=suffix_id)
    all_datasets = convert_jsonl_to_dict(all_datasets)
    return all_datasets

if __name__ == "__main__":

    dataset = Dataset.load_from_disk("/home/export/base/ycsc_lijt1/lijt1/online1/zecheng/data/processed_project")

    tokenizer = transformers.AutoTokenizer.from_pretrained("/home/export/base/ycsc_lijt1/lijt1/online1/zecheng/hf_models/Meta-Llama-3-8B-Instruct")
    training_samples = []
    avg_real_seq_length = 0
    spe_token_id = tokenizer("<|reserved_special_token_0|>", add_special_tokens=False).input_ids[0]

    # processed_data = dataset.map(map_fn, fn_kwargs={"tokenizer": tokenizer, "special_token_id": spe_token_id, "qa_size": 256, "max_embedding_size": 65536, "real_reference_size": 8192, "prefix_id": None, "suffix_id": None}, num_proc=24, load_from_cache_file=False)
    
    """
    final_data = []
    with tqdm(total=len(processed_data)) as pbar:
        for item in processed_data:
            training_samples.extend([sub_ for sub_ in item])
            pbar.update(1)
    import pdb; pdb.set_trace()
    """
    with tqdm(total=len(dataset), desc=f"Initial Avg Length: {avg_real_seq_length}") as pbar:
        for item in dataset:
            all_ref_text = item["all_ref_text"]
            combined_question, final_answer = item["combined_question"], item["final_answer"]
            prefix_q, suffix_q = item["prefix_q"], item["suffix_q"]
            prefix_a, suffix_a = item["prefix_a"], item["suffix_a"]
            # prefix_id, suffix_id = find_index(item["all_ref_ids"], item["prefix_id"], item["suffix_id"])
            prefix_id, suffix_id = None, None
            all_datasets, ref_length = create_covering_position_ipt_data(tokenizer, all_ref_text, combined_question, final_answer, prefix_a, suffix_a, qa_size=256, max_embedding_size=65536, real_reference_size=8192, special_token_id=spe_token_id, prefix_id=prefix_id, suffix_id=suffix_id)
            avg_real_seq_length += ref_length / len(dataset)
            training_samples.extend(all_datasets)
            pbar.set_description(f"Current Avg Seq Length: {avg_real_seq_length:.2f}")
            pbar.update(1)
    print(len(training_samples))
    
    train_sample = training_samples[500:]
    valid_sample = training_samples[:500]
    
    auto_save_data(train_sample, "/home/export/base/ycsc_lijt1/lijt1/online1/zecheng/data/step4_processed_data/train.pkl", show_meta_data=True) 
    auto_save_data(valid_sample, "/home/export/base/ycsc_lijt1/lijt1/online1/zecheng/data/step4_processed_data/valid.pkl", show_meta_data=True) 

    exit()

    df = pd.DataFrame(training_samples)
    
    with tqdm(total=len(df.columns), desc="Converting tensors to numpy") as pbar_outer:
        for column in df.columns:
            if isinstance(df[column][0], torch.Tensor):
                with tqdm(total=len(df[column]), desc=f"Processing {column}") as pbar_inner:

                    def convert_tensor(x):
                        result = x.numpy().tolist()
                        pbar_inner.update(1)
                        return result

                    df[column] = df[column].apply(convert_tensor)
            pbar_outer.update(1)
    import pdb; pdb.set_trace()
    hf_dataset = Dataset.from_pandas(df)
    import pdb; pdb.set_trace()
