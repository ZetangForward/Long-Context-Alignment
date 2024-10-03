import os
import math
import torch
import datasets
import transformers
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence
from torch.utils.data import Dataset
from transformers import Trainer
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM
from torch.distributed import barrier
from datasets import load_dataset, load_from_disk
import numpy as np
from modelzipper.tutils import *
from transformers import AutoConfig


class CustomDataCollator2:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        instructions = [item['instruction'] for item in batch]
        outputs = [item['output'] for item in batch]

        instruct_toks = [self.tokenizer(item, return_tensors="pt").input_ids for item in instructions]
        output_toks = [self.tokenizer(item, return_tensors="pt").input_ids for item in outputs]
        
        input_ids_list = []
        attention_masks_list = []
        labels_list = []

        for inst, out in zip(instruct_toks, output_toks):
            if inst.size(-1) + out.size(-1) > self.max_seq_len:
                remain_seq_length = self.max_seq_len - out.size(-1)
                inst = torch.cat([inst[:, :remain_seq_length // 2], inst[:, -remain_seq_length // 2:]], dim=1)

            # Constructing labels, where non-output tokens are labeled as -100
            labels = torch.full(inst.size(), -100, dtype=torch.long)
            labels = torch.cat([labels, out], dim=1)

            if labels.size(-1) < self.max_seq_len:
                padding_length = self.max_seq_len - labels.size(-1)
                labels = torch.cat([labels, torch.full((1, padding_length), -100, dtype=torch.long)], dim=1)

            # Collecting the processed data
            full_input_ids = torch.cat([inst, out], dim=1)
            # Attention Mask
            attention_mask = torch.ones_like(full_input_ids, dtype=torch.long)
            if full_input_ids.size(-1) < self.max_seq_len:
                padding_length = self.max_seq_len - full_input_ids.size(-1)
                full_input_ids = torch.cat([full_input_ids, torch.zeros((1, padding_length), dtype=torch.long)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.zeros((1, padding_length), dtype=torch.long)], dim=1)

            input_ids_list.append(full_input_ids.squeeze(0))
            attention_masks_list.append(attention_mask.squeeze(0))
            labels_list.append(labels.squeeze(0))
        
        # Stack each list to create a batched tensor
        batch_data = {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_masks_list),
            'labels': torch.stack(labels_list)
        }
        return batch_data


class CustomDataCollator1:
    
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_seq_length = max_seq_length
    
    def wrap_query(self, query, answer):
        chat = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]
        res = self.tokenizer.apply_chat_template(chat, tokenize=False)
        return res
        
    def __call__(self, batch):
        batch_s = [self.wrap_query(item["instruction"], item["output"]) for item in batch]
        wrap_batch = [self.tokenizer(s, padding='max_length', max_length=self.max_seq_length, pad_to_max_length=True, return_tensors="pt", truncation=True) for s in batch_s]
        input_ids = torch.concat([item["input_ids"] for item in wrap_batch], dim=0)
        attention_mask = torch.concat([item["attention_mask"] for item in wrap_batch], dim=0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")
    rope_type: Optional[str] = field(default=None)
    factor: int = field(default=10, metadata={"help": "RoPE factor"})
    max_position_embeddings: int = field(default=10, metadata={"help": "max_position_embeddings"})
    rope_theta: float = field(default=None, metadata={"help": "rope_theta"})
    peft_model_path: Optional[str] = field(default=None)
    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    dataset_path: str = field(default="alpaca")
    model_max_length: int = field(
        default=8192 * 10,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use plain, full-attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )


class CustomTrainier(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, **kwargs):
        super().__init__(
            model=model, 
            args=args, 
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset, 
            tokenizer=tokenizer,
            **kwargs,
        )
 
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=inputs["labels"])
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss 
    
def train():

    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    if model_args.rope_type is not None:
        config.rope_scaling = {"type": model_args.rope_type, "factor": model_args.factor}
    if model_args.rope_theta is not None:
        config.rope_theta = model_args.rope_theta

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    train_dataset = auto_read_data("/data/zecheng/data/LongAlpaca-12k/LongAlpaca-12k.json")
   
    data_collator = CustomDataCollator1(tokenizer, max_seq_length=16384)
  
    if training_args.low_rank_training:
        print("load lora ...")
        if model_args.peft_model_path is not None:
            model = PeftModelForCausalLM.from_pretrained(model, model_args.peft_model_path)
        else:
            if model_args.model_type == "gpt-neox":
                # added `dense` to match with llama as the basic LoRA would only target 'query_key_value'
                targets = ["query_key_value", "dense"]
            else:
                targets=["q_proj", "k_proj", "v_proj", "o_proj"]

            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=targets,
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
            # enable trainable params
        [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]
        model.print_trainable_parameters()
    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing
    
    trainer = CustomTrainier(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
    
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()