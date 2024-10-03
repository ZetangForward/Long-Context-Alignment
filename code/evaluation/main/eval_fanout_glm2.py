import os
import datasets
import json
import torch
from tqdm import tqdm
from typing import Optional, Dict, List
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from accelerate import Accelerator
from transformers import HfArgumentParser
from transformers.utils import logging
from torch.utils.data import DataLoader

from src import ModelArgs, DefaultDataCollator, FileLogger, get_model_and_tokenizer, makedirs
from .longbench_utils import qa_f1_score, rouge_score_needle

logger = logging.get_logger(__name__)


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
    max_new_tokens: int = field(
        default=200,
        metadata={'help': 'Max input length.'}
    )
    max_length: int = field(
        default=31500,
        metadata={'help': 'Max input length.'}
    )
    rouge: str = field(
        default='rouge-l',
    )
    eva_indic: str = field(
        default='f',
    )
    note: str = field(
        default='32k',
    )
    do_sample: bool = False
    


def process_fanout(tokenizer, chat_template, max_length):
    
    def filter_tok_context_length(s, L, tokenizer):
        tok_context = tokenizer(s, return_tensors='pt', add_special_tokens=False).input_ids[0]
        decoded_context = tokenizer.decode(tok_context[:L], skip_special_tokens=True)
        return decoded_context
    
    def _process(data, indices):
        outputs = {'input_ids': [], 'attention_mask': [], "index": []}
        TEMPLATE = "*** BEGIN DATA ***\n\n{context}\n*** END DATA ***\n\n \
                Answer the following question based on the documents above, and output only your answer. \
                If the answer is a list, output one on each line. \n\n[Question]: {question}"
        # QUESTION_TEMPLATE = "<document>\n<title>{title}</title>\n<content>{evidence}</content>\n</document>\n"

        for question, context, index in zip(data['question'], data['all_evidence'], indices):
            evidences = [item for sublist in context for inner_list in sublist for item in inner_list]
            num_evidences = len(evidences) 
            tokenized_template = tokenizer(TEMPLATE+question, truncation=False, return_tensors="pt").input_ids[0]
            all_evi = "\n".join(evidences)
            tokenized_all_evi = tokenizer(all_evi, truncation=False, return_tensors="pt").input_ids[0]
            all_length = len(tokenized_all_evi) + len(tokenized_template) 
            if max_length < (all_length):
                need_filt_length = all_length - max_length 
                per_filt_need = need_filt_length // num_evidences 
                per_evidence_max_length = (max_length-len(tokenized_template)) // num_evidences 
                new_evis = [filter_tok_context_length(s, per_evidence_max_length, tokenizer) for s in evidences] 
                new_context = "\n".join(new_evis)

            else:
                new_context = all_evi
            prompt = TEMPLATE.format(context=new_context, question=question)
            prompt = tokenizer.build_prompt(prompt)

            encoded = tokenizer(prompt)
            encoded.pop('position_ids')

            for k, v in encoded.items():

                outputs[k].append(v)
            outputs["index"].append(index)

        return outputs
    return _process


@torch.no_grad()
def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=args.cpu)
    model, tokenizer = get_model_and_tokenizer(args, device=accelerator.device)

    with accelerator.main_process_first():
        process_fn = process_fanout(
            tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
        )

        raw_dataset = datasets.load_dataset("json", data_files=args.eval_data, cache_dir=args.dataset_cache_dir, split="train")
        
        dataset = raw_dataset.map(process_fn, batched=True, num_proc=32, with_indices=True, remove_columns=raw_dataset.column_names)
        

    result_dir = os.path.join(args.output_dir, args.result_dir)
    result_path = os.path.join(result_dir, f"{'results_' + args.note}.json")

    # dataset = datasets.Dataset.from_pandas(dataset, preserve_index=False)
    data_collator = DefaultDataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=data_collator,
        # only pin memory when no gpu
        pin_memory=not args.cpu,
    )

    if not args.enable_tp:
        # NOTE: prepare model only once
        if len(accelerator._models) == 0:
            model, dataloader = accelerator.prepare(model, dataloader)
            model = accelerator.unwrap_model(model)
        else:
            dataloader = accelerator.prepare(dataloader)
    else:
        # NOTE: prepare dataloader so the data moves to GPU automatically
        dataloader = accelerator.prepare(dataloader)

    indices = []
    preds = []

    for i,x in enumerate(tqdm(dataloader, desc="Generating")):
        index = x.pop("index")[0]
        input_length = x["input_ids"].shape[1]
        print(input_length)

        # NOTE: important to reset memory for every batch
        if hasattr(model, "memory"):
            model.memory.reset()

        output = model.generate(
            **x,
            max_new_tokens=512,
            # min_new_tokens=64,
            num_beams=1,
            do_sample=False,
            temperature=1.0,
            synced_gpus=accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3,
        )

        output = output[:, input_length:]

        print(tokenizer.batch_decode(output, skip_special_tokens=True))
        
        if accelerator.num_processes > 1:
            # pad across device to the same length
            output = accelerator.pad_across_processes(output.contiguous(), pad_index=tokenizer.pad_token_id, dim=1)
            # num_device, max_new_tokens
            output = accelerator.gather_for_metrics(output)
            index = accelerator.gather_for_metrics(index)
        
        output = output.tolist()
        index = index.tolist()

        if accelerator.process_index == 0:
            pred = tokenizer.batch_decode(output, skip_special_tokens=True)
            preds.extend(pred)
            if isinstance(index, list):
                indices.extend(index)
            else:
                # single process
                indices.append(index)
            all_answers = raw_dataset['answer']
            
    if accelerator.process_index == 0:
        f1 = 0
        rouge = 0    
        for pred, answer in zip(preds, all_answers):
            f1 += round(qa_f1_score(pred, answer), 4)
            rouge = rouge_score_needle(pred, answer)
            rouge = rouge[args.rouge][args.eva_indic]
            
        
        f1 /= len(all_answers)
        rouge /= len(all_answers)

        with open(makedirs(result_path), "w", encoding="utf-8") as f:
            f.write(json.dumps(f1, ensure_ascii=False) + "\n")
            f.write(json.dumps(rouge, ensure_ascii=False) + "\n")
            for index, pred in zip(indices, preds):
                sample = raw_dataset[index]
                del sample["all_evidence"]
                sample["pred"] = pred
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        # save config
        metrics = {'rouge': rouge}
        metrics['f1'] = f1
        args.save(os.path.join(result_dir, "config.json"))
        file_logger = FileLogger(makedirs(os.path.join(result_dir, "metrics.log")))
        file_logger.log(metrics, Args=asdict(args))




if __name__ == "__main__":
    main()
