import os
import math
import torch
import transformers
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM
from torch.distributed import barrier
from datasets import load_dataset, load_from_disk
import numpy as np
from scipy.stats import qmc
from transformers import AutoConfig
from modelzipper.tutils import *

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


class CustomDataCollator:
    
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __call__(self, batch):
        # Tokenize inputs
        tokenized_inputs = self.tokenizer(batch, padding="max_length", truncation=True,max_length=self.max_seq_len, return_tensors="pt")
        
        # Rename input_ids in tokenized_labels to labels
        labels = tokenized_inputs["input_ids"].clone()
        
        # Replace padding token id with -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Add labels to the tokenized inputs dictionary
        tokenized_inputs["labels"] = labels
        return tokenized_inputs



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")
    rope_type: Optional[str] = field(default="rope_type")
    factor: int = field(default=10, metadata={"help": "RoPE factor"})
    max_position_embeddings: int = field(default=10, metadata={"help": "max_position_embeddings"})
    rope_theta: float = field(default=10, metadata={"help": "rope_theta"})
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
    tokenizer_max_length: int = field(
        default=4096,
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

    
def train():

    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    # config.max_position_embeddings = model_args.max_position_embeddings

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    train_dataset = load_dataset(training_args.dataset_path)['train']['text']
    data_collator = CustomDataCollator(tokenizer, max_seq_len=16384)
    
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
    
    trainer = Trainer(
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