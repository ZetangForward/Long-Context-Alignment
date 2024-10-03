import os
import torch
from typing import Dict, List, Optional
from datasets import Dataset, load_dataset, load_from_disk, DatasetDict, concatenate_datasets
# from datasets.utils import size_str
from tqdm import tqdm
import numpy as np
import itertools
from functools import partial
from accelerate import PartialState
import multiprocessing
from transformers import AutoTokenizer
import random
import logging
import pandas as pd
from modelzipper import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SparsePosBuilder:
    def __init__(self, dataset_paths: Dict, model_path: str, save_dir: str, max_embedding_size: int, system_prompt_size: int = 0, qa_size: int = None, real_reference_size: int = None, spe_token: int = ' . ', model_provider: str = None, num_valid_samples: int = 200, num_train_samples: int = 10000):
        self.dataset_paths = dataset_paths
        self.raw_datasets = [load_from_disk(dataset_paths[p]) for p in dataset_paths]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.save_dir = save_dir
        self.special_token = spe_token
        self.max_embedding_size = max_embedding_size
        self.system_prompt_size = system_prompt_size
        self.qa_size = qa_size
        self.real_ref_size = real_reference_size  # this is the real size of the reference
        self.real_input_size = system_prompt_size + real_reference_size + qa_size  # this is the real size of the input 
        self.expend_ref_size = max_embedding_size - qa_size - system_prompt_size  # this is the expended size of reference chunk size
        self.num_valid_samples = num_valid_samples
        self.num_train_samples = num_train_samples
        if model_provider.lower() == 'llama2':
            self.SYSTEM_PREFIX = '[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\nBelow is some references. Please read them carefully and answer the following question. <</SYS>>\n\n'
            self.QUESTION_TEMPLATE = 'Please answer the following question according to the references: {question} [/INST]'
            self.ANSWER_TEMPLATE = ' {answer} </s>'
        else:
            self.SYSTEM_PREFIX = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBelow is some references. Please read them carefully and answer the following question.<|eot_id|>'
            self.QUESTION_TEMPLATE = '<|start_header_id|>user<|end_header_id|>\n\nPlease answer the following question according to the references: {question}<|eot_id|>'
            self.ANSWER_TEMPLATE = '<|start_header_id|>assistant<|end_header_id|>\n\nThe answer is: {answer}<|eot_id|><|end_of_text|>'
        
        
    def create_position_ids(self, N, L):
        """sampling N points from L (max_chunk_size space)"""
        if N == L:
            start_pos = 0
        else:
            start_pos = np.random.randint(0, L - N)
        end_pos = start_pos + N
        position_ids = torch.arange(start_pos, end_pos)
        return position_ids 
    

    def create_covering_position_ids(self, N, L):
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


    def auto_padding(self, t: torch.Tensor, length: int, filling_value=-100, return_attention_mask=False):
        """Pad a tensor (t) to a specified length (length) with a given value (filling_value), if return_attention_mask is open, return attention mask"""
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
    

    def create_system_prefix(self):
        """Create system prefix"""
        tok_prefix = self.tokenizer(self.SYSTEM_PREFIX, return_tensors="pt", add_special_tokens=False)
        sys_prefix_ipt_ids, sys_prefix_attn_mask = tok_prefix.input_ids[0], tok_prefix.attention_mask[0]
        system_pos_ids = torch.arange(0, sys_prefix_ipt_ids.size(-1))
        last_extend_pos_id = sys_prefix_ipt_ids.size(-1)
        if self.system_prompt_size == 0:  # not defined
            self.system_prompt_size = sys_prefix_ipt_ids.size(-1)
            self.real_input_size += self.system_prompt_size
            self.expend_ref_size -= self.system_prompt_size
        return sys_prefix_ipt_ids, sys_prefix_attn_mask, system_pos_ids, last_extend_pos_id
    

    def combine_fn_random_sample(self, lst, max_candidates=2, max_combination=16):
        """Random sample and combine the sublsts in the lst, with max_candidates and max_combination as the upper bound"""
        trimmed_lists = [random.sample(sublst, min(len(sublst), max_candidates)) for sublst in lst]
        all_combinations = list(itertools.islice(itertools.product(*trimmed_lists), max_combination))
        random.shuffle(all_combinations)
        return [torch.cat(combination) for combination in all_combinations[:max_combination]]


    def combine_fn_order_sample(self, lst, max_candidates=8):
        """Sample from each sublist"""
        combinations = []
        for _ in range(max_candidates):
            combination = [random.choice(sublst) for sublst in lst]
            combinations.append(combination)
        return [torch.cat(combination, dim=0) for combination in combinations]
    
    
    def cut_str_length(self, s: str, max_length: int):
        """Cut the string to the maximum length allowed by the tokenizer."""
        input_ids = self.tokenizer(s, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if input_ids.size(-1) > max_length:
            input_ids = input_ids[:max_length]
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)
    

    def post_process_sparse_subsets(
        self, 
        sys_prefix_ipt_ids: torch.Tensor, 
        sys_prefix_attn_mask: torch.Tensor, 
        system_pos_ids: torch.Tensor, 
        sparse_padded_ref_pos_ids_cand: List[torch.Tensor], 
        sparse_padded_ref_inp_ids: torch.Tensor, 
        sparse_padded_ref_attn_mask: torch.Tensor, 
        sparse_qa: Dict[str, torch.Tensor]
    ):
        """  
        Post-process Sparse Input Data Here !!!
        Process one input sample in sparse format, 
        with different position ids for the reference chunk
        """
        sparse_sets = []
        q_pos_ids_cand, cho_a_padded_pos_ids_cand, rej_a1_padded_pos_ids_cand, rej_a2_padded_pos_ids_cand = sparse_qa['question'][2], sparse_qa['chosen_answer'][2], sparse_qa['rejected_answer1'][2], sparse_qa['rejected_answer2'][2]
        
        for i, sparse_paded_ref_pos_id in enumerate(sparse_padded_ref_pos_ids_cand):
            concatenated_batch = {}
            q_pos_ids, cho_a_padded_pos_ids, rej_a1_padded_pos_ids, rej_a2_padded_pos_ids = q_pos_ids_cand[i], cho_a_padded_pos_ids_cand[i], rej_a1_padded_pos_ids_cand[i], rej_a2_padded_pos_ids_cand[i]
            q_input_ids = torch.concat([sys_prefix_ipt_ids, sparse_padded_ref_inp_ids, sparse_qa['question'][0]], dim=0)
            q_attention_mask = torch.concat([sys_prefix_attn_mask, sparse_padded_ref_attn_mask, sparse_qa['question'][1]], dim=0)
            q_position_ids = torch.concat([system_pos_ids, sparse_paded_ref_pos_id, q_pos_ids], dim=0)
            referece_question_labels = torch.full((1, q_attention_mask.size(-1)), -100)[0]
            concatenated_batch['chosen'] = {
                'input_ids': torch.concat([q_input_ids, sparse_qa['chosen_answer'][0]]), 
                'attention_mask': torch.concat([q_attention_mask, sparse_qa['chosen_answer'][1]]), 
                'position_ids': torch.concat([q_position_ids, cho_a_padded_pos_ids]), 
                'labels': torch.concat([referece_question_labels, sparse_qa['chosen_answer'][3]], dim=0),
            }
            concatenated_batch['reject_1'] = {
                'input_ids': torch.concat([q_input_ids, sparse_qa['rejected_answer1'][0]]), 
                'attention_mask': torch.concat([q_attention_mask, sparse_qa['rejected_answer1'][1]]), 
                'position_ids': torch.concat([q_position_ids, rej_a1_padded_pos_ids]), 
                'labels': torch.concat([referece_question_labels, sparse_qa['rejected_answer1'][3]], dim=0),
            }
            concatenated_batch['reject_2'] = {
                'input_ids': torch.concat([q_input_ids, sparse_qa['rejected_answer2'][0]]), 
                'attention_mask': torch.concat([q_attention_mask, sparse_qa['rejected_answer2'][1]]), 
                'position_ids': torch.concat([q_position_ids, rej_a2_padded_pos_ids]), 
                'labels': torch.concat([referece_question_labels, sparse_qa['rejected_answer2'][3]], dim=0),
            }
            sparse_sets.append(concatenated_batch)
        return sparse_sets
    
    
    def post_process_cont_subsets(
        self, 
        sys_prefix_ipt_ids: torch.Tensor, 
        sys_prefix_attn_mask: torch.Tensor, 
        system_pos_ids: torch.Tensor, 
        cont_padded_ref_pos_ids_cand: torch.Tensor, 
        cont_padded_ref_inp_ids: torch.Tensor, 
        cont_padded_ref_attn_mask: torch.Tensor,
        cont_qa: Dict[str, torch.Tensor]
    ):
        """  
        Post-process Sparse Input Data Here !!!
        Process one input sample in sparse format, 
        with different position ids for the reference chunk
        Attention: We pad to the max sequence length here, 
        since continuous chunk does not fit the self.max_embedding_size
        """
        cont_sets = []
        q_pos_ids_cand, cho_a_padded_pos_ids_cand, rej_a1_padded_pos_ids_cand, rej_a2_padded_pos_ids_cand = cont_qa['question'][2], cont_qa['chosen_answer'][2], cont_qa['rejected_answer1'][2], cont_qa['rejected_answer2'][2]
        
        for i, sparse_paded_ref_pos_id in enumerate(cont_padded_ref_pos_ids_cand):
            concatenated_batch = {}
            q_pos_ids, cho_a_padded_pos_ids, rej_a1_padded_pos_ids, rej_a2_padded_pos_ids = q_pos_ids_cand[i], cho_a_padded_pos_ids_cand[i], rej_a1_padded_pos_ids_cand[i], rej_a2_padded_pos_ids_cand[i]
            q_input_ids = torch.concat([sys_prefix_ipt_ids, cont_padded_ref_inp_ids, cont_qa['question'][0]], dim=0)
            q_attention_mask = torch.concat([sys_prefix_attn_mask, cont_padded_ref_attn_mask, cont_qa['question'][1]], dim=0)
            q_position_ids = torch.concat([system_pos_ids, sparse_paded_ref_pos_id, q_pos_ids], dim=0)
            referece_question_labels = torch.full((1, q_attention_mask.size(-1)), -100)[0]
            chosen_padding_size = self.real_input_size - q_input_ids.size(-1) - cont_qa['chosen_answer'][0].size(-1)
            reject_1_padding_size = self.real_input_size - q_input_ids.size(-1) - cont_qa['rejected_answer1'][0].size(-1)
            reject_2_padding_size = self.real_input_size - q_input_ids.size(-1) - cont_qa['rejected_answer2'][0].size(-1)
            
            concatenated_batch['chosen'] = {
                'input_ids': torch.concat([q_input_ids, cont_qa['chosen_answer'][0], torch.zeros(chosen_padding_size, dtype=torch.int)], dim=0), 
                'attention_mask': torch.concat([q_attention_mask, cont_qa['chosen_answer'][1], torch.zeros(chosen_padding_size, dtype=torch.int)], dim=0), 
                'position_ids': torch.concat([q_position_ids, cho_a_padded_pos_ids, torch.zeros(chosen_padding_size, dtype=torch.int)], dim=0), 
                'labels': torch.concat([referece_question_labels, cont_qa['chosen_answer'][3], torch.full((chosen_padding_size,), -100, dtype=torch.int)], dim=0) 
            }
            concatenated_batch['reject_1'] = {
                'input_ids': torch.concat([q_input_ids, cont_qa['rejected_answer1'][0], torch.zeros(reject_1_padding_size, dtype=torch.int)], dim=0),
                'attention_mask': torch.concat([q_attention_mask, cont_qa['rejected_answer1'][1], torch.zeros(reject_1_padding_size, dtype=torch.int)], dim=0),
                'position_ids': torch.concat([q_position_ids, rej_a1_padded_pos_ids, torch.zeros(reject_1_padding_size, dtype=torch.int)], dim=0),
                'labels': torch.concat([referece_question_labels, cont_qa['rejected_answer1'][3], torch.full((reject_1_padding_size,), -100, dtype=torch.int)], dim=0)
            }
            concatenated_batch['reject_2'] = {
                'input_ids': torch.concat([q_input_ids, cont_qa['rejected_answer2'][0], torch.zeros(reject_2_padding_size, dtype=torch.int)], dim=0), 
                'attention_mask': torch.concat([q_attention_mask, cont_qa['rejected_answer2'][1], torch.zeros(reject_2_padding_size, dtype=torch.int)], dim=0),
                'position_ids': torch.concat([q_position_ids, rej_a2_padded_pos_ids, torch.zeros(reject_2_padding_size, dtype=torch.int)], dim=0),  
                'labels': torch.concat([referece_question_labels, cont_qa['rejected_answer2'][3], torch.full((reject_2_padding_size,), -100, dtype=torch.int)], dim=0)
            }
            cont_sets.append(concatenated_batch)
        return cont_sets
    
    
    def create_cover_pos_ipt_data(self, all_refs: List[str], cho_q: str, cho_a: str, rej_a1: str, rej_a2: str,  add_continue_chunk: bool = False):
        """
        Process one input sample, which will have different position ids for the reference chunk
        """
        
        sys_prefix_ipt_ids, sys_prefix_attn_mask, system_pos_ids, last_extend_pos_id = self.create_system_prefix()
        one_sample_sets = []
        statistic_pos_distribution = torch.zeros(self.expend_ref_size, dtype=torch.int) # must init after the system prefix is created !!!
        
        # Create Sparse Input Data Here !!!
        res = self.create_ref_chunk(all_refs, add_continue_chunk)
        sparse_padded_ref_pos_ids_cand, sparse_padded_ref_inp_ids, sparse_padded_ref_attn_mask, pos_idx_cnt, last_extend_pos_ids = res["sparse_ref_chunk"]
        sparse_qa = self.create_qa(cho_q, cho_a, rej_a1, rej_a2, last_extend_pos_ids)
        statistic_pos_distribution += pos_idx_cnt
        sparse_sets = self.post_process_sparse_subsets(
            sys_prefix_ipt_ids, sys_prefix_attn_mask, system_pos_ids, 
            sparse_padded_ref_pos_ids_cand, sparse_padded_ref_inp_ids, sparse_padded_ref_attn_mask, 
            sparse_qa
        )
        auto_check = [self.check_data(sparse_sets[0][k]) for k in sparse_sets[0]]
        if all([e[0] for e in auto_check]):
            one_sample_sets.extend(sparse_sets)
        else:
            print([e[1] for e in auto_check])
            raise ValueError("Sparse Data Check Failed!")
        
        # Create Continuous Input Data Here !!!
        if add_continue_chunk:
            cont_padded_ref_pos_ids_cand, cont_padded_ref_inp_ids, cont_padded_ref_attn_mask, pos_idx_cnt,last_extend_pos_ids = res["continue_ref_chunk"]
            cont_qa = self.create_qa(cho_q, cho_a, rej_a1, rej_a2, last_extend_pos_ids)  
            statistic_pos_distribution += pos_idx_cnt
            cont_sets = self.post_process_cont_subsets(
                sys_prefix_ipt_ids, sys_prefix_attn_mask, system_pos_ids, 
                cont_padded_ref_pos_ids_cand, cont_padded_ref_inp_ids, cont_padded_ref_attn_mask, 
                cont_qa
            )
            auto_check = [self.check_data(cont_sets[0][k]) for k in cont_sets[0]]
            if all([e[0] for e in auto_check]):
                one_sample_sets.extend(cont_sets)
            else:
                print([e[1] for e in auto_check])
                raise ValueError("Sparse Data Check Failed!")

        return one_sample_sets, statistic_pos_distribution
    
    
    def create_qa(self, cho_q: str, cho_a: str, rej_a1: str, rej_a2: str, last_extend_pos_ids: int):
        """
        Return:
            question:
                padded_q_inp_ids: torch.Tensor
                padded_q_attn_mask: torch.Tensor
                q_pos_ids_cand: torch.Tensor
            chosen_answer:
                cho_a_padded_inp_ids: torch.Tensor
                cho_a_padded_attn_mask: torch.Tensor
                cho_a_padded_pos_ids_cand: torch.Tensor
                cho_a_padded_labels: torch.Tensor
            rejected_answer1:
                rej_a1_padded_inp_ids: torch.Tensor
                rej_a1_padded_attn_mask: torch.Tensor
                rej_a1_padded_pos_ids_cand: torch.Tensor
                rej_a1_padded_labels: torch.Tensor
            rejected_answer2:
                rej_a2_padded_inp_ids: torch.Tensor
                rej_a2_padded_attn_mask: torch.Tensor
                rej_a2_padded_pos_ids_cand: torch.Tensor
                rej_a2_padded_labels: torch.Tensor
        """
        # Create Question
        question = self.QUESTION_TEMPLATE.format(question=cho_q)
        tok_question = self.tokenizer(question, return_tensors="pt", add_special_tokens=False).input_ids[0]
        padded_q_inp_ids, padded_q_attn_mask = self.auto_padding(tok_question, tok_question.size(-1), filling_value=0, return_attention_mask=True)
        q_pos_ids_cands = [torch.arange(0, tok_question.size(-1)) + last_pos_id + 1 for last_pos_id in last_extend_pos_ids]
        last_extend_pos_ids = [q_pos_ids.max() for q_pos_ids in q_pos_ids_cands]

        # Create Chosen / Rejected Answers / and their labels
        cho_a = self.ANSWER_TEMPLATE.format(answer=cho_a)
        rej_a1 = self.ANSWER_TEMPLATE.format(answer=rej_a1)
        rej_a2 = self.ANSWER_TEMPLATE.format(answer=rej_a2)
        cho_a_inp_ids = self.tokenizer(cho_a, return_tensors="pt", add_special_tokens=False).input_ids[0]
        rej_a1_inp_ids = self.tokenizer(rej_a1, return_tensors="pt", add_special_tokens=False).input_ids[0]
        rej_a2_inp_ids = self.tokenizer(rej_a2, return_tensors="pt", add_special_tokens=False).input_ids[0]

        cho_a_padded_inp_ids, cho_a_padded_attn_mask = self.auto_padding(cho_a_inp_ids, self.qa_size-padded_q_inp_ids.size(-1), filling_value=0, return_attention_mask=True)
        cho_a_padded_labels = self.auto_padding(cho_a_inp_ids, self.qa_size-padded_q_inp_ids.size(-1), filling_value=-100)
        cho_a_padded_pos_ids_cand = [torch.arange(0, cho_a_padded_inp_ids.size(-1)) + last_pos_id + 1 for last_pos_id in last_extend_pos_ids] # padding will mask the extra position ids

        rej_a1_padded_inp_ids, rej_a1_padded_attn_mask = self.auto_padding(rej_a1_inp_ids, self.qa_size-padded_q_inp_ids.size(-1), filling_value=0, return_attention_mask=True)
        rej_a1_padded_labels = self.auto_padding(rej_a1_inp_ids, self.qa_size-padded_q_inp_ids.size(-1), filling_value=-100)
        rej_a1_padded_pos_ids_cand = [torch.arange(0, rej_a1_padded_inp_ids.size(-1)) + last_pos_id + 1 for last_pos_id in last_extend_pos_ids] # padding will mask the extra position ids
        
        rej_a2_padded_inp_ids, rej_a2_padded_attn_mask = self.auto_padding(rej_a2_inp_ids, self.qa_size-padded_q_inp_ids.size(-1), filling_value=0, return_attention_mask=True)
        rej_a2_padded_labels = self.auto_padding(rej_a2_inp_ids, self.qa_size-padded_q_inp_ids.size(-1), filling_value=-100)
        rej_a2_padded_pos_ids_cand = [torch.arange(0, rej_a2_padded_inp_ids.size(-1)) + last_pos_id + 1 for last_pos_id in last_extend_pos_ids] # padding will mask the extra position ids
    
        return {
            'question': [padded_q_inp_ids, padded_q_attn_mask, q_pos_ids_cands],
            'chosen_answer': [cho_a_padded_inp_ids, cho_a_padded_attn_mask, cho_a_padded_pos_ids_cand, cho_a_padded_labels],
            'rejected_answer1': [rej_a1_padded_inp_ids, rej_a1_padded_attn_mask, rej_a1_padded_pos_ids_cand, rej_a1_padded_labels],
            'rejected_answer2': [rej_a2_padded_inp_ids, rej_a2_padded_attn_mask, rej_a2_padded_pos_ids_cand, rej_a2_padded_labels],
        }
    
    
    def create_ref_chunk(self, all_refs: List[str], add_continue_chunk: bool = False):
        """create sparse chunk position ids, padded ref input ids, padded ref attention mask, and all special token positions"""
        sparse_pos_padded_ref_pos_ids_cand, sparse_padded_ref_input_ids, sparse_padded_ref_attention_mask, pos_idx_cnt, last_extend_pos_ids = self.create_sparse_chunked_reference(all_refs, max_combination=4, return_statistic=True)
        res = {
                "sparse_ref_chunk": (
                    sparse_pos_padded_ref_pos_ids_cand, 
                    sparse_padded_ref_input_ids, 
                    sparse_padded_ref_attention_mask,
                    pos_idx_cnt, 
                    last_extend_pos_ids
                )
            }
        if add_continue_chunk: # Create continuous chunked reference
            cont_pos_padded_ref_pos_ids_cand, cont_padded_ref_input_ids, cont_padded_ref_attn_mask, pos_idx_cnt, last_extend_pos_ids = self.create_continue_chunked_reference(all_refs, return_statistic=True)
            res["continue_ref_chunk"] = (
                cont_pos_padded_ref_pos_ids_cand, 
                cont_padded_ref_input_ids,
                cont_padded_ref_attn_mask, 
                pos_idx_cnt,
                last_extend_pos_ids
            )
        return res
    
    
    def create_sparse_chunked_reference(self, all_refs: List[str], max_combination: int = None, return_statistic: bool = False):
        """
        Return:
            sparse_pos_padded_ref_pos_ids_cand: List[torch.Tensor], 
            sparse_padded_ref_input_ids: torch.Tensor, 
            sparse_padded_ref_attention_mask: torch.Tensor,
            pos_idx_cnt [Optional]: torch.Tensor, 
            last_extend_pos_id: int
        """
        
        all_refs = [s + self.special_token for s in all_refs]
        chunk_ref_size = self.real_ref_size // len(all_refs)
        truncted_refer_tok_lst = [self.cut_str_length(ref, chunk_ref_size) for ref in all_refs]
        truncted_refer_tok_lst = [self.tokenizer(item, return_tensors="pt", add_special_tokens=False).input_ids[0] for item in truncted_refer_tok_lst]
        
        expend_chunk_ref_size = self.expend_ref_size // (len(all_refs) + 1)  # prevent the last chunk to overflow the max seq length
        positional_chunks = torch.arange(self.system_prompt_size, self.expend_ref_size, expend_chunk_ref_size)

        begin_positional_chunks = positional_chunks[:-1]
        all_chunk_pos_lst = []  # there is some empty space at the end, requires to pad
        
        max_chunks_upper_bound = 4
        
        for _, item in enumerate(truncted_refer_tok_lst):
            chunk_token_pos_lst = self.create_covering_position_ids(item.size(-1), expend_chunk_ref_size)
            all_chunk_pos_lst.append(chunk_token_pos_lst)
            max_chunks_upper_bound = max_chunks_upper_bound * len(chunk_token_pos_lst)

        if max_combination is None:
            max_combination = min(max_chunks_upper_bound, 8)  # we take 8 as the minimum value
        
        padded_refer_tok_lst = [self.auto_padding(item, chunk_ref_size, filling_value=0, return_attention_mask=True) for item in truncted_refer_tok_lst]
        padded_ref_tok_ids, padded_ref_attn_mask = [item[0] for item in padded_refer_tok_lst], [item[1] for item in padded_refer_tok_lst]
        for item, bos_pos in zip(all_chunk_pos_lst, begin_positional_chunks):
            for sub_item in item:
                sub_item += bos_pos # add bos_position to each chunk positions
        padded_chunk_pos_lst = [[self.auto_padding(sub_item, chunk_ref_size, filling_value=0, return_attention_mask=False) for sub_item in item] for item in all_chunk_pos_lst]
        sparse_pos_padded_ref_pos_ids_cand = self.combine_fn_order_sample(padded_chunk_pos_lst, max_candidates=max_combination)
        sparse_padded_ref_input_ids = torch.concatenate(padded_ref_tok_ids, dim=0)
        sparse_padded_ref_attn_mask = torch.concatenate(padded_ref_attn_mask, dim=0)
   
        pos_idx_cnt = None
        if return_statistic:
            pos_idx_cnt = torch.zeros(self.expend_ref_size, dtype=torch.int)
            for item in sparse_pos_padded_ref_pos_ids_cand:
                pos_idx_cnt[item] += 1
        
        last_extend_pos_ids = [e.max() for e in sparse_pos_padded_ref_pos_ids_cand]
        return (
            sparse_pos_padded_ref_pos_ids_cand, 
            sparse_padded_ref_input_ids, 
            sparse_padded_ref_attn_mask, 
            pos_idx_cnt,
            last_extend_pos_ids, # last extend position id
        ) 
        
        
    def create_continue_chunked_reference(self, all_refs: List[str], return_statistic: bool = False):
        """
        Return:
            cont_pos_padded_ref_pos_ids_cand: List[torch.Tensor], 
            cont_padded_ref_input_ids: torch.Tensor, 
            cont_padded_ref_attn_mask: torch.Tensor,
            pos_idx_cnt [Optional]: torch.Tensor, 
            last_extend_pos_ids: List[int], 
        """
        
        last_extend_pos_ids = []
        ref_str = self.special_token.join(all_refs)
        ref_str = self.cut_str_length(ref_str, self.real_ref_size)
        cont_chunk_tok = self.tokenizer(ref_str, return_tensors="pt", add_special_tokens=False).input_ids[0]
        cont_padded_ref_input_ids, cont_padded_ref_attn_mask = self.auto_padding(cont_chunk_tok, self.real_ref_size, filling_value=0, return_attention_mask=True)
        pos_chunks = torch.arange(self.system_prompt_size, self.max_embedding_size - self.qa_size, cont_padded_ref_attn_mask.sum())
        cont_pos_padded_ref_pos_ids_cand = []

        for begin_pos in pos_chunks:
            if begin_pos + cont_padded_ref_attn_mask.sum() > self.max_embedding_size - self.qa_size:
                break
            position_ids = torch.arange(begin_pos, begin_pos + cont_padded_ref_attn_mask.sum())
            padded_position_ids = self.auto_padding(position_ids, self.real_ref_size, filling_value=0, return_attention_mask=False)
            cont_pos_padded_ref_pos_ids_cand.append(padded_position_ids)
            last_extend_pos_ids.append(begin_pos + self.real_ref_size)  # last extended position ids
        
        pos_idx_cnt = None
        if return_statistic:
            pos_idx_cnt = torch.zeros(self.expend_ref_size, dtype=torch.int)
            for item in cont_pos_padded_ref_pos_ids_cand:
                pos_idx_cnt[item] += 1
        
        return (
            cont_pos_padded_ref_pos_ids_cand,
            cont_padded_ref_input_ids, 
            cont_padded_ref_attn_mask,
            pos_idx_cnt,
            last_extend_pos_ids
        )


    def flatten_dataset(self, d):
        feature_names = d.column_names
        processed_features = {name: [] for name in feature_names}
        with tqdm(total=len(d), desc="Flattening the dataset ...") as pbar:
            for example in d:
                for name in feature_names:
                    processed_features[name].extend(example[name])
                pbar.update(1)
        return Dataset.from_dict(processed_features)

    def map_fn(self, batch, add_continual_chunk=False):
        failed_dict = {'chosen_input_ids': None, 'chosen_attention_mask': None, 'chosen_position_ids': None, 'chosen_labels': None, 'reject_1_input_ids': None, 'reject_1_attention_mask': None, 'reject_1_position_ids': None, 'reject_1_labels': None, 'reject_2_input_ids': None, 'reject_2_attention_mask': None, 'reject_2_position_ids': None, 'reject_2_labels': None}

        if batch['judge_scores'] < 1:
            return failed_dict
        try:
            # if len(batch['all_ref_text']) > 4:
            #     merged_list = []
            #     all_ref_text = batch["all_ref_text"]
            #     for i in range(0, len(all_ref_text), 2):
            #         if i + 1 < len(all_ref_text):
            #             merged = all_ref_text[i] + ' ' + all_ref_text[i+1]
            #             merged_list.append(merged)
            # batch["all_ref_text"] = merged_list 
            one_sample_sets, statistic_pos_distribution = self.create_cover_pos_ipt_data(
                all_refs=batch["all_ref_text"], cho_q=batch["combined_question"], cho_a=batch["final_answer"], rej_a1=batch["prefix_a"], rej_a2=batch["siffix_a"], add_continue_chunk=add_continual_chunk
            )
            converted_one_sample_sets = dict(
                chosen_input_ids = [], chosen_attention_mask = [], chosen_position_ids = [], chosen_labels = [],
                reject_1_input_ids = [], reject_1_attention_mask = [], reject_1_position_ids = [], reject_1_labels = [],
                reject_2_input_ids = [], reject_2_attention_mask = [], reject_2_position_ids = [], reject_2_labels = [],
            )
            for gen_item in one_sample_sets:
                converted_one_sample_sets['chosen_input_ids'].append(gen_item['chosen']['input_ids'])
                converted_one_sample_sets['chosen_attention_mask'].append(gen_item['chosen']['attention_mask'])
                converted_one_sample_sets['chosen_position_ids'].append(gen_item['chosen']['position_ids'])
                converted_one_sample_sets['chosen_labels'].append(gen_item['chosen']['labels'])
                converted_one_sample_sets['reject_1_input_ids'].append(gen_item['reject_1']['input_ids'])
                converted_one_sample_sets['reject_1_attention_mask'].append(gen_item['reject_1']['attention_mask'])
                converted_one_sample_sets['reject_1_position_ids'].append(gen_item['reject_1']['position_ids'])
                converted_one_sample_sets['reject_1_labels'].append(gen_item['reject_1']['labels'])
                converted_one_sample_sets['reject_2_input_ids'].append(gen_item['reject_2']['input_ids'])
                converted_one_sample_sets['reject_2_attention_mask'].append(gen_item['reject_2']['attention_mask'])
                converted_one_sample_sets['reject_2_position_ids'].append(gen_item['reject_2']['position_ids'])
                converted_one_sample_sets['reject_2_labels'].append(gen_item['reject_2']['labels'])
            return converted_one_sample_sets
        except:
            return failed_dict


    def filter_none(self, example):
        
        return all(value is not None for value in example.values())


    def build(self, add_continue_chunk: bool = True, num_proc=1):
        '''Process one sample'''
        all_datasets = Dataset.from_dict(
            dict(
                chosen_input_ids = [], chosen_attention_mask = [], chosen_position_ids = [], chosen_labels = [],
                reject_1_input_ids = [], reject_1_attention_mask = [], reject_1_position_ids = [], reject_1_labels = [],
                reject_2_input_ids = [], reject_2_attention_mask = [], reject_2_position_ids = [], reject_2_labels = [],
            )
        )
        for i, sub_raw_dataset in enumerate(self.raw_datasets):  
            dataset_name = list(self.dataset_paths.keys())[i]
            if len(sub_raw_dataset) > 5000:
                sub_raw_dataset = sub_raw_dataset.shuffle(seed=42).select(range(5000))
            with tqdm(total=len(sub_raw_dataset), desc=f"Processing {dataset_name} dataset, has {len(all_datasets['chosen_input_ids'])} samples ...") as pbar:
                custom_map_fn = partial(self.map_fn, add_continual_chunk=add_continue_chunk)
                sub_res = sub_raw_dataset.map(custom_map_fn, num_proc=num_proc, remove_columns=['all_ref_text', 'combined_question', 'final_answer', 'prefix_a', 'siffix_a', 'judge_scores', 'label', 'judger_preds'])
                filter_sub_res = sub_res.filter(self.filter_none, num_proc=num_proc)
                del sub_res
                logger.info('flatten the datasets, it may cost some time ...')
                filter_sub_res = self.flatten_dataset(filter_sub_res)
                logger.info('flatten ending, merge to final datasets ...')
                all_datasets = concatenate_datasets([all_datasets, filter_sub_res])
                pbar.update(1)
        all_datasets = all_datasets.shuffle(seed=42)   
        all_datasets = all_datasets.select(range(self.num_valid_samples + self.num_train_samples))
        self.post_process(all_datasets, self.save_dir) 

        ''' NOTE: this is old code which can only process with single process
        for j, item in enumerate(sub_raw_dataset):   
            try:
                pbar.update(1)
                if item['judge_scores'] < 1:
                    continue
                if len(item['all_ref_text']) > 4:
                    merged_list = []
                    all_ref_text = item["all_ref_text"]
                    for i in range(0, len(all_ref_text), 2):
                        if i + 1 < len(all_ref_text):
                            merged = all_ref_text[i] + ' ' + all_ref_text[i+1]
                            merged_list.append(merged)
                item["all_ref_text"] = merged_list 
                one_sample_sets, statistic_pos_distribution = self.create_cover_pos_ipt_data(
                    all_refs=item["all_ref_text"], cho_q=item["combined_question"], cho_a=item["final_answer"],
                    rej_a1=item["prefix_a"], rej_a2=item["siffix_a"], add_continue_chunk=add_continue_chunk
                )
                for gen_item in one_sample_sets:
                    all_datassets['chosen']['input_ids'].append(gen_item['chosen']['input_ids'])
                    all_datassets['chosen']['attention_mask'].append(gen_item['chosen']['attention_mask'])
                    all_datassets['chosen']['position_ids'].append(gen_item['chosen']['position_ids'])
                    all_datassets['chosen']['labels'].append(gen_item['chosen']['labels'])
                    all_datassets['reject_1']['input_ids'].append(gen_item['reject_1']['input_ids'])
                    all_datassets['reject_1']['attention_mask'].append(gen_item['reject_1']['attention_mask'])
                    all_datassets['reject_1']['position_ids'].append(gen_item['reject_1']['position_ids'])
                    all_datassets['reject_1']['labels'].append(gen_item['reject_1']['labels'])
                    all_datassets['reject_2']['input_ids'].append(gen_item['reject_2']['input_ids'])
                    all_datassets['reject_2']['attention_mask'].append(gen_item['reject_2']['attention_mask'])
                    all_datassets['reject_2']['position_ids'].append(gen_item['reject_2']['position_ids'])
                    all_datassets['reject_2']['labels'].append(gen_item['reject_2']['labels'])
                statistic_pos_distributions.extend(statistic_pos_distribution)
            except:
                logger.error(f"Error in processing the sample id: {j} from {dataset_name} dataset")
        '''


    def post_process(self, all_datasets, save_dir):
        """Post-process the generated all datasets"""
        # split training and validation datasets
        # train_samples, valid_samples = all_datasets[num_valid_samples:], all_datasets[:num_valid_samples]
        # train_datasets, valid_datasets = Dataset.from_dict(train_samples), Dataset.from_dict(valid_samples)
        # dataset_dict = DatasetDict({'train': train_datasets, 'valid': valid_datasets}) 
        train_valid_split = all_datasets.train_test_split(test_size=self.num_valid_samples)
        # print(size_str(train_valid_split))
        train_valid_split.save_to_disk(self.save_dir, num_shards=32, num_proc=32)
        logger.info(f"Saving finished, the processed datasets are located in {save_dir}")
  

    def check_data(self, data):
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        position_ids = data['position_ids']
        labels = data['labels']
        errors = []
        masked_input_ids = input_ids[attention_mask == 1]
        masked_position_ids = position_ids[attention_mask == 1]
        
        # if not torch.all(masked_input_ids != 0):  # ! can be encoded as 0
        #     errors.append("input_ids invalid at masked positions")
        
        if not torch.all(input_ids[attention_mask == 0] == 0):
            errors.append("input_ids padding must be zero")
        
        # if not torch.all(position_ids[attention_mask == 0] == 0):  
            # errors.append("position_ids padding must be zero")

        if not torch.all(masked_position_ids[:-1] < masked_position_ids[1:]):
            errors.append("position_ids should be in ascending order without duplicates")

        valid_labels = labels[labels != -100]
        corresponding_input_ids = input_ids[labels != -100]
        
        if not torch.equal(valid_labels, corresponding_input_ids):
            errors.append("labels for loss calculation do not match corresponding input_ids")

        if errors:
            return False, "; ".join(errors)
        else:
            return True, "All checks passed"
    
    
if __name__ == "__main__":
    dir_path = '/nvme/zecheng/data/iclr2025/llama3-80k-train-data/long-llm-score/chunk_16_size_1024'
    score_dataset_paths = {
        'gpt-bio_book': os.path.join(dir_path, 'gpt-bio_book'),
        'gpt-multi_detail_book': os.path.join(dir_path, 'gpt-multi_detail_book'),
        'gpt-multi_detail_paper_long': os.path.join(dir_path, 'gpt-multi_detail_paper_long'),
        'gpt-multi_detail_paper_short': os.path.join(dir_path, 'gpt-multi_detail_paper_short'),
        # 'gpt-one_detail_paper': os.path.join(dir_path, 'gpt-one_detail_paper'),
        'longalpaca-train': os.path.join(dir_path, 'longalpaca-train'),
    }
    
    model_path='/data/zecheng/hf_models/longchat-7b-v1.5-32k'
    save_path = '/nvme/zecheng/data/iclr2025/llama2-train-data/dpo_data/chunk_16_size_1024'
    
    real_reference_size, qa_size, max_embedding_size = 1024 * 16, 1000, 65536 # 实际训练长度卡在16K上下(System Prompt长度需要动态判断)

    builder = SparsePosBuilder(
        score_dataset_paths, model_path, save_path, 
        max_embedding_size=max_embedding_size,
        qa_size=qa_size, 
        real_reference_size=real_reference_size,
        model_provider='llama2',
        num_valid_samples = 200,
        num_train_samples = 15000,
    )

    # builder = SparsePosBuilder(
    #     score_dataset_paths, model_path, save_path, 
    #     max_embedding_size=max_embedding_size,
    #     qa_size=qa_size, 
    #     real_reference_size=real_reference_size,
    #     model_provider = 'llama3',
    #     num_valid_samples = 200,
    #     num_train_samples = 10000,
    # )
    
    bt = time.time()
    res = builder.build(add_continue_chunk=True, num_proc=128)
    print_c(f"Time: {time.time() - bt} seconds")

    