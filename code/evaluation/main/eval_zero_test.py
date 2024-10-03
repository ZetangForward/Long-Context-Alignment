import json
import os
import sys
from datetime import datetime
import random

import numpy as np
import torch
from fire import Fire
from transformers import set_seed as hf_set_seed
from src import ModelArgs, get_model_and_tokenizer
import os
import datasets
import json
import torch
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field
from accelerate import Accelerator
from transformers import HfArgumentParser
from .longbench_utils import DATASET2MAXNEWTOKENS

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class Args(ModelArgs):
    eval_data: str = field(
        default="long-llm:longbench/test.json",
        metadata={'help': 'The evaluation json data path.'}
    )
    output_dir: str = field(
        default="data/results/longbench/",
        metadata={'help': 'The base directory for saving results and logs.'}
    )
    result_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'The directory relative to output_dir for saving results.'}
    )
    dataset_names: str = field(
        default=None,
        metadata={'help': 'Which dataset to evaluate?'}
    )

    max_length: int = field(
        default=31500,
        metadata={'help': 'Max input length.'}
    )
    load_result: bool = field(
        default=False,
        metadata={'help': 'Load result from saved files?'}
    )

    do_sample: bool = False
    

def process_model_input(tokenizer, example, max_tokens, device):
    tokenized_input_full = tokenizer(example["input"], return_tensors="pt").input_ids.to(device)
    if tokenized_input_full.shape[1] <= max_tokens:
        return tokenized_input_full

    seperator_and_query_text = example['truncation_seperator'] + example["input"][example['query_start_index']:]
    tokenized_seperator_and_query = tokenizer(seperator_and_query_text, return_tensors="pt").input_ids.to(device)
    input_without_query = example['input'][:example['query_start_index']]
    tokenized_input_without_query = tokenizer(input_without_query, return_tensors="pt").input_ids.to(device)
    tokenized_input_without_query = tokenized_input_without_query[:, :max_tokens - tokenized_seperator_and_query.shape[1]]

    tokenized_input = torch.cat([tokenized_input_without_query, tokenized_seperator_and_query], dim=1)
    return tokenized_input


def main(max_examples_per_task=-1):
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]
    seed = 43
    random.seed(seed)
    np.random.seed(seed)
    hf_set_seed(seed)
    print("Params:")
    model_name = args.model_name_or_path
    print(f"model: {model_name}")
    generations_dir = os.path.join(os.path.join(args.output_dir, args.result_dir))
    print(f"generations_dir: {generations_dir}")
    print(f"max_examples_per_task: {max_examples_per_task}")
    print("=" * 50)
    time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    print(f"time as start: {time}")


    print("Loading tokenizer")
    accelerator = Accelerator(cpu=args.cpu)
    model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)

    print(f"Loading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_input_length = args.max_length
    print(f"{model} model loaded!, device:{model.device}")

    print("Will write to:", generations_dir)
    os.makedirs(generations_dir, exist_ok=True)

    dataset = args.eval_data

    print(f"Processing {dataset}")
    time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    print(f"time as start {dataset}: {time}")
    print(f"Loading {dataset}")

    data = datasets.load_dataset("json", data_files=dataset, cache_dir=args.dataset_cache_dir, split="train")
    print(f"Loaded {dataset}")
    
    generations = dict()
    generations_all = dict()
    gen_res = []
    for i, example in enumerate(tqdm(data)):
        if 0 < max_examples_per_task == i:
            print(f"Reached {max_examples_per_task} for {dataset}. Breaking")
            break
        model_input = process_model_input(tokenizer, example, max_input_length, device)
        max_new_tokens = DATASET2MAXNEWTOKENS[args.dataset_names]
        prediction_token_ids = model.generate(model_input,
                                                max_new_tokens=max_new_tokens,
                                                do_sample=False,
                                                top_p=0,
                                                top_k=0,
                                                temperature=1)
        
        input_length = model_input.size(1)
        predicted_text = tokenizer.decode(prediction_token_ids[:, input_length:][0], skip_special_tokens=True)
        predicted_text_all = tokenizer.decode(prediction_token_ids[0], skip_special_tokens=True)

        generations[example["id"]] = predicted_text
        generations_all[example["id"]] = predicted_text_all
        

    out_file_path = os.path.join(generations_dir, f"preds_{args.dataset_names}.json")
    with open(out_file_path, 'w') as f_out:
        json.dump(generations, f_out, indent=4)

    out_file_path_all = os.path.join(generations_dir, f"preds_{args.dataset_names}_all.json")
    with open(out_file_path_all, 'w') as f_out:
        json.dump(generations_all, f_out, indent=4)

    print(f"Done generating {len(generations)} examples from {dataset}")
    time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    print(f"time at end: {time}")
    print(f"Look for predictions in {generations_dir}")


if __name__ == '__main__':
    main()
