## 使用Deepseek模型对摘出来的文档进行筛选，筛掉非文本的部分

from modelzipper.tutils import *
import re
from pprint import pprint
import itertools
import random
import transformers
from datasets import Dataset, concatenate_datasets, DatasetDict, load_from_disk, load_dataset
import pandas as pd
import numpy as np
from openai import OpenAI
import threading
from http import HTTPStatus
import itertools


### warning ###
os.environ['DEEPSEEK_API_KEY'] = "sk-99ed89d2c4dd4fb381ba80f638149d6c"

API_KEY = os.environ.get("DEEPSEEK_API_KEY")

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

REFERENCE_Q_A_PAIR = "##Reference: {Reference}\n##Question {Question}\n##Answer is {Answer}##. Remember, just return a single word ``NULL'' or ``Yes''"

SYSTEM_TEMPLATE ="""User will provide a question-answer pair based on the reference.

First check if the reference is natural text (not containing a large amount of code, numbers, and URLs) to prevent the invalidity of the problem. 

If the reference is non natural text, please return a single word ``NULL'' directly without further judgment. 

Else just return a single word ``YES''.
"""


def extract_qa_pairs(text):
    sp_text = text.split("####")
    if len(sp_text) < 9:
        return None
    q1, a1, q2, a2 = sp_text[2].strip(), sp_text[4].strip(), sp_text[6].strip(), sp_text[8].strip()   
    res = [{"question": q1, "answer": a1}, {"question": q2, "answer": a2}]
    return res


def combine_data(data, chunk_num):
    combined_data = []
    for i in range(0, len(data), chunk_num):
        combined_data.append(data[i:i+chunk_num])
    return combined_data


def single_item(item):
    item["filtered_response"] = []
    for qa_pair in item['qa_pairs']:
        reference_qa_pair = REFERENCE_Q_A_PAIR.format(
            Reference=item['reference'],
            Question=qa_pair['question'],
            Answer=qa_pair['answer'],
        )
        response = single_response(reference_qa_pair)
        item["filtered_response"].append(response)
    return item


def single_response(query):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user", "content": query},
        ],
        stream=False,
    )
    return response.choices[0].message.content

def mp_single_response(items, save_path, log_path, thread_id):
    with open(save_path, "a") as save_f, open(log_path, "a") as log_f:
        with tqdm(total=len(items), desc=f"Processing thread {thread_id}") as pbar:
            for item in items:
                id = item["id"]
                try:
                    response = single_item(item)
                    save_f.write(json.dumps(response) + "\n")
                except:
                    continue  # pass the bad data
                log_f.write(str(id) + "\n")
                pbar.update(1)

def allocate_ids(remain_process_data, total_processes):
    avg_length = len(remain_process_data) // total_processes
    remainder = len(remain_process_data) % total_processes
    allocated_ids = []
    start_index = 0
    
    for i in range(total_processes):
        end_index = start_index + avg_length + (1 if i < remainder else 0)
        allocated_ids.append(remain_process_data[start_index:end_index])
        start_index = end_index
    
    return allocated_ids

if __name__ == "__main__":

    dir_path = "/vepfs/wcf/G/zecheng/data/SlimPajama-6B"
    filter_dir_path = "/vepfs/wcf/G/zecheng/data/SlimPajama-6B/filtering_deepseek"
    if os.path.exists(os.path.join(dir_path, "all_qa_pairs.jsonl")):
        processed_data = auto_read_data(os.path.join(dir_path, "all_qa_pairs.jsonl"))
    else:
        # pre-preprocess all created data
        files = auto_read_dir(dir_path, file_prefix="generated_QA_pairs_thread", file_suffix=".jsonl")
        all_data = [auto_read_data(file, mute=True) for file in files]
        all_data = [item for sublist in all_data for item in sublist]
        processed_data = []
        for item in all_data:
            qa_pairs = extract_qa_pairs(item['QA_pairs'])
            if qa_pairs is not None:
                ref = item['reference']
                ref = ref if isinstance(ref, str) else ref[1]
                processed_data.append({"reference": ref, "qa_pairs": qa_pairs})

        # allocate each sample with an identified id
        for i, item in enumerate(processed_data):
            item['id'] = i
        auto_save_data(processed_data, os.path.join(dir_path, "all_qa_pairs.jsonl"))
    

    # exclude already processed data
    all_processed_ids = list(range(len(processed_data))) 
    already_processed_data = [auto_read_data(f, mute=True) for f in auto_read_dir(filter_dir_path, file_suffix=".jsonl")]
    already_processed_data = list(itertools.chain(*already_processed_data))
    if len(already_processed_data) > 0:
        already_processed_ids = set([item['id'] for item in already_processed_data])
    else:
        already_processed_ids = set()
    remain_process_data = list(set(all_processed_ids) - already_processed_ids)

    # allocate the remain_process_data
    num_threads = 10
    allocated_ids = allocate_ids(remain_process_data, num_threads)

    # then filter the reference QA pairs (mp)
    threads = []
    for thread_id in range(num_threads):
        cur_allocate_ids = allocated_ids[thread_id]
        cur_processed_data = [item for item in processed_data if item['id'] in cur_allocate_ids]
        save_path = os.path.join(filter_dir_path, f"thread_{thread_id}_filter.jsonl")
        log_path = os.path.join(filter_dir_path, f"thread_{thread_id}_filter.log")
        thread = threading.Thread(target=mp_single_response, args=(cur_processed_data, save_path, log_path, thread_id))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    