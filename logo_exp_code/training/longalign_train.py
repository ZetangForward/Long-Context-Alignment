import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments

class LMDataset(torch.utils.data.Dataset):
    def __init__(self, filepath):
        self.input_ids, self.labels = self.process_data(filepath)

    def process_data(self, filepath):
        input_ids = torch.from_numpy(np.load(os.path.join(filepath, 'inputs.npy')))
        labels = torch.from_numpy(np.load(os.path.join(filepath, 'labels.npy')))
        return input_ids, labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return self.input_ids.size(0)

class LMSortDataset(torch.utils.data.Dataset):
    def __init__(self, filepath):
        self.input_ids, self.labels = self.process_data(filepath)
    
    def process_data(self, filepath):
        input_ids = torch.from_numpy(np.load(os.path.join(filepath, 'inputs_sort.npy')))
        labels = torch.from_numpy(np.load(os.path.join(filepath, 'labels_sort.npy')))
        return input_ids, labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return self.input_ids.size(0)

class LMPackDataset(torch.utils.data.Dataset):
    def __init__(self, filepath):
        self.input_ids, self.attention_masks, self.labels, self.weights, self.nums = self.process_data(filepath)
        self.num_gpus = torch.cuda.device_count()
        
    def process_data(self, filepath):
        input_ids = torch.from_numpy(np.load(os.path.join(filepath, 'inputs_pack.npy')))
        labels = torch.from_numpy(np.load(os.path.join(filepath, 'labels_pack.npy')))
        weights = torch.from_numpy(np.load(os.path.join(filepath, 'weights_pack.npy')))
        attention_masks = json.load(open(os.path.join(filepath, 'attention_masks_pack.json')))
        num_gpus = torch.cuda.device_count()
        l = (input_ids.size(0) // num_gpus) * num_gpus
        input_ids, labels, weights, attention_masks = input_ids[:l, :], labels[:l, :], weights[:l, :], attention_masks[:l]
        nums = [weights[i*num_gpus:(i+1)*num_gpus, :].sum() for i in range(l//num_gpus)]
        return input_ids, attention_masks, labels, weights, nums

    def __getitem__(self, idx):
        if idx < 32: # reduce GPU memory usage during first few steps
            max_length_tmp = 32768
            attention_mask_tmp = []
            for pos in self.attention_masks[idx]:
                if pos < max_length_tmp:
                    attention_mask_tmp.append(pos)
            attention_mask_tmp.append(max_length_tmp)
            return {
                'input_ids': self.input_ids[idx, :max_length_tmp],
                'attention_mask': torch.tensor(attention_mask_tmp, dtype=torch.int32),
                'labels': (self.labels[idx, :max_length_tmp], self.weights[idx, :max_length_tmp]*2, self.nums[idx//self.num_gpus])
            }
        else:
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.int32),
                'labels': (self.labels[idx], self.weights[idx], self.nums[idx//self.num_gpus])
            }

    def __len__(self):
        return self.input_ids.size(0)
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="THUDM/LongAlign-6B-64k-base")
    pack_loss: bool = field(default=False)

@dataclass
class DataArguments:
    train_file: str = field(default=None, metadata={"help": "Path to the training data."})
    validation_file: str = field(default=None, metadata={"help": "Path to the training data."})
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    prompt_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    response_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    batch_method: str = field(default="naive")

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

@dataclass
class DataCollatorForLMDataset(object):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key].unsqueeze(0) for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)
        eos_indices = input_ids.argmin(dim=1) - 1
        max_position = eos_indices.max()
        if max_position < 0:
            return dict(
                input_ids=input_ids,
                labels=labels
            )
        return dict(
            input_ids=input_ids[:, :max_position+1],
            labels=labels[:, :max_position+1]
        )

@dataclass
class DataCollatorForLMPackDataset(object):

    def __call__(self, instances):
        input_ids, attention_masks = tuple([instance[key].unsqueeze(0) for instance in instances] for key in ["input_ids", "attention_mask"])
        batch_seq_num = instances[0]["labels"][2]
        labels = ([instance["labels"][0].unsqueeze(0) for instance in instances], [instance["labels"][1].unsqueeze(0) for instance in instances])
        input_ids = torch.cat(input_ids, dim=0)
        labels = (torch.cat(labels[0], dim=0), torch.cat(labels[1], dim=0))
        labels = (labels[0], labels[1] * torch.cuda.device_count() / batch_seq_num)
        max_length = input_ids.shape[1]
        attention_mask = attention_masks[0].squeeze()
        acc_length = max_length
        for new_attention_mask in attention_masks[1:]:
            new_attention_mask = new_attention_mask.squeeze()
            attention_mask = torch.cat([attention_mask, new_attention_mask[1:]+acc_length], dim=0)
            acc_length += max_length
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
class TrainerNoShuffle(Trainer):
    def __init__(
        self,
        model = None,
        args: TrainingArguments = None,
        data_collator = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer = None,
        model_init = None,
        compute_metrics = None,
        callbacks = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]: # disable shuffling
        return SequentialSampler(self.train_dataset)

def make_supervised_data_module(data_args) -> Dict:
    if data_args.batch_method == "naive":
        train_dataset = LMDataset(data_args.train_file)
        data_collator = DataCollatorForLMDataset()
    elif data_args.batch_method == "pack":
        train_dataset = LMPackDataset(data_args.train_file)
        data_collator = DataCollatorForLMPackDataset()
    elif data_args.batch_method == "sort":
        train_dataset = LMSortDataset(data_args.train_file)
        data_collator = DataCollatorForLMDataset()
    return dict(train_dataset=train_dataset, data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        torch_dtype=torch.bfloat16, 
        
    )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                                  trust_remote_code=True)
    if model_args.pack_loss:
        model.pack_loss = True
    data_module = make_supervised_data_module(data_args=data_args)

    trainer = TrainerNoShuffle(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        **data_module
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model()

if __name__ == "__main__":
    train()