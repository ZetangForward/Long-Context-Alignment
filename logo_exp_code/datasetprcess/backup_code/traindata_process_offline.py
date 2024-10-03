import os
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional
from datasets import Dataset, load_dataset, load_from_disk, DatasetDict
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    HfArgumentParser, 
    set_seed, 
    AutoConfig, 
    TrainingArguments,
)
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


max_chunk_size = 256
max_position_embeddings = 4096 * 16
max_qa_size = 1024


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


def create_random_position_ipt_data(feature):
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
    
    chosen_attention_mask = torch.concat([padded_reference_attention_mask, question_chosen_ans_attention_mask], dim=1)
    rejected_attention_mask = torch.concat([padded_reference_attention_mask, question_rejected_ans_attention_mask], dim=1)
    concatenated_batch["concatenated_attention_mask"] = torch.concat([chosen_attention_mask, rejected_attention_mask], dim=0)

    chosen_labels = torch.concat([reference_labels, question_chosen_labels], dim=1)
    rejected_labels = torch.concat([reference_labels, question_rejected_labels], dim=1)
    concatenated_batch["concatenated_labels"] = torch.concat([chosen_labels, rejected_labels], dim=0)

    chosen_position_ids = torch.concat([padded_position_ids, question_chosen_ans_position_ids.unsqueeze(0)], dim=1)
    rejected_position_ids = torch.concat([padded_position_ids, question_rejected_ans_position_ids.unsqueeze(0)], dim=1)
    concatenated_batch["position_ids"] = torch.concat([chosen_position_ids, rejected_position_ids], dim=0)
    return concatenated_batch

cache_path = "/vepfs/wcf/G/zecheng/data/SlimPajama-6B/dpo_data/hf_data_64_cache"
save_path = "/vepfs/wcf/G/zecheng/data/SlimPajama-6B/dpo_data/hf_data_64_cache_v2"
all_data = load_from_disk(cache_path)
print(all_data)
all_data = all_data.map(create_random_position_ipt_data, num_proc=64, remove_columns=['reference_list', 'question', 'chosen', 'rejected', 'refer_tok_lst', 'question_tok', 'chosen_tok_ans', 'reject_tok_ans'])
# all_data.save_to_disk(save_path, num_proc=4)