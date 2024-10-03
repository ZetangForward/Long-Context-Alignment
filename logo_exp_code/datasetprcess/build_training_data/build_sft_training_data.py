import os
import torch
from typing import Dict, List, Optional
from datasets import Dataset, load_dataset, load_from_disk, DatasetDict
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
    
    def __init__(self, dataset_paths: Dict, model_path: str, save_dir: str, max_embedding_size: int, system_prompt_size: int = 0, qa_size: int = None, real_reference_size: int = None, spe_token: int = ' . ', model_provider: str = None, num_valid_samples: int = 200):
        self.dataset_paths = dataset_paths
        self.raw_datasets = [auto_read_data(dataset_paths[p]) for p in dataset_paths]
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
        if model_provider.lower() == 'film':
            self.SYSTEM_PREFIX = '[INST] Below is a context and an instruction. Based on the information provided in the context, write a response for the instruction. ### Context:\n'
            self.QUESTION_TEMPLATE = '### Instruction: {question} [/INST]'
            self.ANSWER_TEMPLATE = '{answer}'
        if model_provider.lower() == 'llama2':
            self.SYSTEM_PREFIX = 'Below is some references. Please read it carefully and answer the following question.'
            self.QUESTION_TEMPLATE = 'Please answer the following question according to the references: {question}'
            self.ANSWER_TEMPLATE = 'The answer is: {answer}'
        else:
            self.SYSTEM_PREFIX = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nBelow is some references. Please read it carefully and answer the following question.<|eot_id|>'
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
        q_pos_ids_cand, cho_a_padded_pos_ids_cand = cont_qa['question'][2], cont_qa['chosen_answer'][2]
        
        for i, sparse_paded_ref_pos_id in enumerate(cont_padded_ref_pos_ids_cand):
            concatenated_batch = {}
            q_pos_ids, cho_a_padded_pos_ids = q_pos_ids_cand[i], cho_a_padded_pos_ids_cand[i]
            q_input_ids = torch.concat([sys_prefix_ipt_ids, cont_padded_ref_inp_ids, cont_qa['question'][0]], dim=0)
            q_attention_mask = torch.concat([sys_prefix_attn_mask, cont_padded_ref_attn_mask, cont_qa['question'][1]], dim=0)
            q_position_ids = torch.concat([system_pos_ids, sparse_paded_ref_pos_id, q_pos_ids], dim=0)
            referece_question_labels = torch.full((1, q_attention_mask.size(-1)), -100)[0]
            chosen_padding_size = self.real_input_size - q_input_ids.size(-1) - cont_qa['chosen_answer'][0].size(-1)
            
            concatenated_batch['chosen'] = {
                'input_ids': torch.concat([q_input_ids, cont_qa['chosen_answer'][0], torch.zeros(chosen_padding_size, dtype=torch.int)], dim=0), 
                'attention_mask': torch.concat([q_attention_mask, cont_qa['chosen_answer'][1], torch.zeros(chosen_padding_size, dtype=torch.int)], dim=0), 
                'position_ids': torch.concat([q_position_ids, cho_a_padded_pos_ids, torch.zeros(chosen_padding_size, dtype=torch.int)], dim=0), 
                'labels': torch.concat([referece_question_labels, cont_qa['chosen_answer'][3], torch.full((chosen_padding_size,), -100, dtype=torch.int)], dim=0) 
            }
            cont_sets.append(concatenated_batch)
        return cont_sets
    
    
    def create_cover_pos_ipt_data(self, all_refs: List[str], cho_q: str, cho_a: str):
        """
        Process one input sample, which will have different position ids for the reference chunk
        """
        
        sys_prefix_ipt_ids, sys_prefix_attn_mask, system_pos_ids, last_extend_pos_id = self.create_system_prefix()
        one_sample_sets = []
        statistic_pos_distribution = torch.zeros(self.expend_ref_size, dtype=torch.int) # must init after the system prefix is created !!!
        
        # Create Sparse Input Data Here !!!
        cont_padded_ref_pos_ids_cand, cont_padded_ref_inp_ids, cont_padded_ref_attn_mask, pos_idx_cnt,last_extend_pos_ids = self.create_ref_chunk(all_refs)
       
        cont_qa = self.create_qa(cho_q, cho_a, last_extend_pos_ids)  
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
    
    
    def create_qa(self, cho_q: str, cho_a: str, last_extend_pos_ids: int):
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
        """
        # Create Question
        question = self.QUESTION_TEMPLATE.format(question=cho_q)
        tok_question = self.tokenizer(question, return_tensors="pt", add_special_tokens=False).input_ids[0]
        padded_q_inp_ids, padded_q_attn_mask = self.auto_padding(tok_question, tok_question.size(-1), filling_value=0, return_attention_mask=True)
        q_pos_ids_cands = [torch.arange(0, tok_question.size(-1)) + last_pos_id + 1 for last_pos_id in last_extend_pos_ids]
        last_extend_pos_ids = [q_pos_ids.max() for q_pos_ids in q_pos_ids_cands]

        # Create Chosen / Rejected Answers / and their labels
        cho_a = self.ANSWER_TEMPLATE.format(answer=cho_a)
        cho_a_inp_ids = self.tokenizer(cho_a, return_tensors="pt", add_special_tokens=False).input_ids[0]

        cho_a_padded_inp_ids, cho_a_padded_attn_mask = self.auto_padding(cho_a_inp_ids, self.qa_size-padded_q_inp_ids.size(-1), filling_value=0, return_attention_mask=True)
        cho_a_padded_labels = self.auto_padding(cho_a_inp_ids, self.qa_size-padded_q_inp_ids.size(-1), filling_value=-100)
        cho_a_padded_pos_ids_cand = [torch.arange(0, cho_a_padded_inp_ids.size(-1)) + last_pos_id + 1 for last_pos_id in last_extend_pos_ids] # padding will mask the extra position ids
    
        return {
            'question': [padded_q_inp_ids, padded_q_attn_mask, q_pos_ids_cands],
            'chosen_answer': [cho_a_padded_inp_ids, cho_a_padded_attn_mask, cho_a_padded_pos_ids_cand, cho_a_padded_labels]
        }
    
    
    def create_ref_chunk(self, all_refs: List[str], add_continue_chunk: bool = False):
        """create continual chunk position ids, padded ref input ids, padded ref attention mask, and all special token positions"""
        
        cont_pos_padded_ref_pos_ids_cand, cont_padded_ref_input_ids, cont_padded_ref_attn_mask, pos_idx_cnt, last_extend_pos_ids = self.create_continue_chunked_reference(all_refs, return_statistic=True)
        return (
            cont_pos_padded_ref_pos_ids_cand, 
            cont_padded_ref_input_ids,
            cont_padded_ref_attn_mask, 
            pos_idx_cnt,
            last_extend_pos_ids
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


    def build(self, add_continue_chunk: bool = True):
        """Process one sample"""
        
        all_datassets, statistic_pos_distributions = [], []
        for i, sub_raw_dataset in enumerate(self.raw_datasets):  
            dataset_name = list(self.dataset_paths.keys())[i]
            with tqdm(total=len(sub_raw_dataset), desc=f"Processing {dataset_name} dataset") as pbar:
                for j, item in enumerate(sub_raw_dataset):   
                    try:
                        all_ref_ids = self.tokenizer(item['context'])['input_ids']
                        merged_list = []
                        for i in range(0, 4):
                            st = i * len(all_ref_ids) // 4
                            if i == 3:
                                ed = len(all_ref_ids)
                            else:
                                ed = st + len(all_ref_ids) // 4
                            merged_list.append(all_ref_ids[st: ed])
                        
                        merged_list = [self.tokenizer.decode(item) for item in merged_list]
                        
                        one_sample_sets, statistic_pos_distribution = self.create_cover_pos_ipt_data(all_refs=merged_list, cho_q=item["input"], cho_a=item["answers"])
                        
                        all_datassets.extend(one_sample_sets)
                        statistic_pos_distributions.extend(statistic_pos_distribution)
                    except:
                        logger.error(f"Error in processing the sample id: {j} from {dataset_name} dataset")
                        
                    pbar.update(1)
                    
        random.shuffle(all_datassets)
        logger.info(f"Total number of samples: {len(all_datassets)}")
        
        self.post_process(all_datassets, self.save_dir, self.num_valid_samples)
        
        return statistic_pos_distributions

    
    def post_process(self, all_datasets, save_dir, num_valid_samples=200):
        """Post-process the generated all datasets"""
        # split training and validation datasets
        train_samples, valid_samples = all_datasets[num_valid_samples:], all_datasets[:num_valid_samples]
        
        try:
            # post-process the data into df format
            logger.info(f"Saving the processed datasets in {save_dir}")
            
            auto_save_data(train_samples, os.path.join(self.save_dir, 'train.pkl'), show_meta_data=True)
            auto_save_data(valid_samples, os.path.join(self.save_dir, 'valid.pkl'), show_meta_data=True)
            
            logger.info(f"Saving finished")
        except:
            logger.error("Error in saving the processed datasets")
            raise ValueError("Error in saving the processed datasets")
        

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
    
    dataset_paths = {
        'qmsum_e': '/vepfs/wcf/G/zecheng/data/long_bench/qmsum_e.jsonl',
        'gov_report_e': '/vepfs/wcf/G/zecheng/data/long_bench/gov_report_e.jsonl',
        'multi_news_e': '/vepfs/wcf/G/zecheng/data/long_bench/multi_news_e.jsonl'
    }
    
    model_path = "/vepfs/wcf/G/zecheng/ckpt/v3"
    save_path = "/vepfs/wcf/G/zecheng/data/v3-aug"
    real_reference_size, qa_size, max_embedding_size = 14000, 1000, 65536 # 实际训练长度卡在16K上下(System Prompt长度需要动态判断)

    tester = SparsePosBuilder(
        dataset_paths, model_path, save_path, 
        max_embedding_size=max_embedding_size,
        qa_size=qa_size, 
        real_reference_size=real_reference_size,
        model_provider = 'llama3',
        num_valid_samples = 0,
    )
    
    bt = time.time()
    res = tester.build()
    print_c(f"Time: {time.time() - bt} seconds")

    