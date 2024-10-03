import itertools
import random
import re
import numpy as np
import pandas as pd
import torch
from typing import List, Set, Tuple, Dict, Optional
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.tokenize import sent_tokenize
import spacy
from functools import partial
from multiprocessing import Pool
from multiprocessing import Process, Queue
from modelzipper.tutils import *
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp
from itertools import repeat
from datasets import set_caching_enabled
set_caching_enabled(False)

spacy_model = spacy.load("en_core_web_lg")
tokenizer = transformers.AutoTokenizer.from_pretrained('/data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged', use_fast=True)

class DataFilter:

    def __init__(self, dataset_dir, dataset_names, model_id, output_path, save_chunk_nums=12, chunk_size=512, min_ne_overlap=3, num_process=1):
        self.dataset_dir = dataset_dir
        self.dataset_names = dataset_names
        self.model_id = model_id
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.save_chunk_nums = save_chunk_nums
        self.min_ne_overlap = min_ne_overlap
        self.num_process = num_process
        self.init_filter()

    def init_filter(self):
        print(f'Initializing filter with model {self.model_id}')
        print(f'Loading datasets from {self.dataset_dir}')
        self.datasets = {}
        with tqdm(total=len(self.dataset_names), desc='Loading datasets') as pbar:
            for dataset_name in self.dataset_names:
                dataset_path = f'{self.dataset_dir}/{dataset_name}'
                self.datasets[dataset_name] = auto_read_data(dataset_path)
                pbar.update(1)
        print('Loading spacy model')
        

    def chunk_text(self, s: str, chunk_size: int=1024) -> List[str]:
        '''
        chunk text according to the chunk_size and tokenizer
        return: List[str]
        '''
        tok_s = tokenizer(s, return_tensors='pt').input_ids[0]
        tok_chunks = [tok_s[i:i+chunk_size] for i in range(0, len(tok_s), chunk_size)]
        return [tokenizer.decode(chunk) for chunk in tok_chunks]


    def get_ne_from_s(self, s) -> Set[str]:
        '''Example: 
        s="What was the cause of the teenager's death in the motorcycle accident? The teenager, John Lister, was killed in a high-speed crash."
        verb = ['kill']
        ents = ['John Lister']
        none = ['cause', 'teenager', 'death', 'motorcycle', 'accident', 'teenager', 'speed', 'crash']
        '''
        doc = spacy_model(s)
        verb = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        ents = [chunk.text for chunk in doc.ents]
        none = [token.lemma_ for token in doc if token.pos_ == "NOUN"]
        return set(verb + ents + none)


    def judge_ne_overlap(self, critical_paths: List[Dict], min_ne_overlap: int = 3) -> bool:
        '''
        Judge if the critical paths have enough overlap
        '''
        all_overlaps = [path['overlap'] for path in critical_paths]
        if min(all_overlaps) >= min_ne_overlap:
            return True
        return False


    def process_sample(self, q: str, a: str, s: str, chunk_size: int=512, save_chunk_nums: int=12, min_ne_overlap: int=3) -> Optional[Dict]:
        '''
        Extract Critical Paths from s, according to the question q and answer a
        '''
        q_ne, a_ne = self.get_ne_from_s(q), self.get_ne_from_s(a)
        s_chunks = self.chunk_text(s, chunk_size)
        union_nes = q_ne.union(a_ne)
        critical_chunks = []
        for i, chunk in enumerate(s_chunks):
            chunk_ne = self.get_ne_from_s(chunk)
            overlap = len(union_nes.intersection(chunk_ne))
            critical_chunks.append({'chunk_id': i, 'chunk': chunk, 'overlap': overlap}) 
        sorted_critical_chunks = sorted(critical_chunks, key=lambda x: x['overlap'], reverse=True)[:save_chunk_nums]
        if self.judge_ne_overlap(sorted_critical_chunks, min_ne_overlap):
            # final_critical_chunks = sorted(sorted_critical_chunks, key=lambda x: x['chunk_id'])
            # final_critical_chunks = [chunk['chunk'] for chunk in final_critical_chunks]
            return {'question': q, 'answer': a, 'context': sorted_critical_chunks}
        return None


    def mp_process_dataset(self, dataset_name: str, content: List[Dict]) -> Dataset:
        # 创建一个Dataset对象
        dataset = Dataset.from_dict({"content": content, "dataset_name": [dataset_name] * len(content)})
        print(f'process data with {self.num_process} processes ...')
        
        processed_dataset = dataset.map( 
            self.process_item,
            remove_columns=dataset.column_names,
            batched=False,
            num_proc=self.num_process,
            desc="Processing dataset",
        )
        return processed_dataset


    def process_item(self, batch):
        processed_items = []
        item, dataset_name = batch['content']['conversations'], batch['dataset_name']
        if dataset_name == 'longalpaca/train.json':
            all_splits = item[0]['content'].split('\n')
            context = '\n'.join(all_splits[2:-2]).strip()
            all_questions = [{'content': all_splits[-1]}]
            all_answers = [{'content': item[1]['content']}]
        else:
            context = '\n\n'.join(item[0]['content'].split('\n\n')[:-1]).strip()
            all_questions, all_answers = item[2::2], item[3::2]
        for q, a in zip(all_questions, all_answers):
            processed_item = self.process_sample(q['content'], a['content'], context, self.chunk_size, self.save_chunk_nums, self.min_ne_overlap)
            if processed_item:
                processed_items.append(processed_item)
        
        if len(processed_items) == 0: # 如果processed_items为空，返回空列表
            return {
                'answer': None,
                'context': None,
                'question': None
            }
               
        return {
            'answer': [item['answer'] for item in processed_items],
            'context': [item['context'] for item in processed_items],
            'question': [item['question'] for item in processed_items]
        }


    def process_dataset(self, content: List[Dict]) -> List[Dict]:
        processed_data = []
        with tqdm(total=len(content), desc='process datasets ...') as pbar1:
            for item in content:
                item = item['conversations']
                context = '\n\n'.join(item[0]['content'].split('\n\n')[:-1]).strip()
                all_questions, all_answers = item[2::2], item[3::2]
                
                with tqdm(total=len(all_questions), desc=f'Processing one single sample, which has {len(all_questions)} queries ...') as pbar2:
                    for q, a in zip(all_questions, all_answers):
                        processed_item = self.process_sample(q['content'], a['content'], context, self.chunk_size, self.save_chunk_nums, self.min_ne_overlap)
                        if processed_item:
                            processed_data.append(processed_item)
                        pbar2.update(1)
                
                pbar1.update(1)
        return processed_data
    

    def begin_process(self):
        '''
        Begin to process
        '''
        # all_processed_data = {}
        for dataset_name in self.dataset_names:
            print(f'Processing dataset {dataset_name}, which has {len(self.datasets[dataset_name])} samples.')
            processed_data = self.mp_process_dataset(dataset_name, self.datasets[dataset_name])

            def is_not_none(example):
                return all(value is not None for value in example.values())
            processed_data = processed_data.filter(is_not_none)
            save_dir = f'{self.output_path}/{dataset_name.split("/")[0]}-{dataset_name.split("/")[-1].split(".")[0]}'
            processed_data.save_to_disk(save_dir)


if __name__ == '__main__':
    dataset_dir = '/data/zecheng/data/llama3-80k-train-data/long-llm'
    dataset_names = [
        'longalpaca/train.json',
        # 'gpt/one_detail_book.train.64K.json',
        'gpt/one_detail_paper.train.64K.json',
        'gpt/multi_detail_book.train.json',
        'gpt/multi_detail_paper_short.train.json',
        'gpt/multi_detail_paper_long.train.json',
        'gpt/bio_book.train.json',
        # 'redpajama/train.json[5000]',
    ]
    print("os.cpu_count()", os.cpu_count())
    data_filter = DataFilter(
        dataset_dir, dataset_names, 
        model_id='/data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged',
        output_path='/nvme/zecheng/data/iclr2025/llama3-80k-train-data/long-llm-filtered/chunk_16_size_1024',
        save_chunk_nums=16, chunk_size=1024, min_ne_overlap=3, num_process=92,
    )

    data_filter.begin_process()
    # auto_save_data(process_res, '/data/zecheng/data/llama3-80k-train-data/long-llm-filtered/rush.pkl')
