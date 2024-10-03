import os
import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM
from transformers import (AutoTokenizer, HfArgumentParser, set_seed, AutoConfig)
from dpo_trainer import DPOTrainer
from typing import Union
from modelzipper.tutils import *
import torch.nn.functional as F
from arguments import ScriptArguments, DPOConfig
from llama import LlamaForCausalLM
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers import Trainer


class CustomDataCollator:

    def inner_wrap_batch(self, batch_data, batch_id):
        input_ids = torch.tensor(batch_data[batch_id]["input_ids"])
        attention_mask = torch.tensor(batch_data[batch_id]["attention_mask"])
        position_ids = torch.tensor(batch_data[batch_id]["position_ids"])
        chosen_answer_input_ids = batch_data[batch_id]["chosen_answer"]["input_ids"]
        chosen_answer_attention_mask = batch_data[batch_id]["chosen_answer"]["attention_mask"]
        chosen_answer_position_ids = batch_data[batch_id]["chosen_answer"]["position_ids"]

        chosen_input_ids = torch.concatenate([input_ids, chosen_answer_input_ids], dim=0).unsqueeze(0)
        chosen_attention_mask = torch.concatenate([attention_mask, chosen_answer_attention_mask], dim=0).unsqueeze(0)
        chosen_position_ids = torch.concatenate([position_ids, chosen_answer_position_ids], dim=0).unsqueeze(0)
        chosen_labels = torch.tensor(batch_data[batch_id]["chosen_answer"]["labels"]).unsqueeze(0)
        
        return chosen_input_ids, chosen_attention_mask, chosen_position_ids, chosen_labels

    def __call__(self, batch):
        wrap_batch = {}

        if len(batch) == 1:
            chosen_input_ids, chosen_attention_mask, chosen_position_ids, chosen_labels = self.inner_wrap_batch(batch, 0)

            wrap_batch["input_ids"] = chosen_input_ids
            wrap_batch["attention_mask"] = chosen_attention_mask
            wrap_batch["position_ids"] = chosen_position_ids
            wrap_batch["labels"] = chosen_labels
            
        else:
            chosen_input_ids_lst, chosen_attention_mask_lst, chosen_position_ids_lst, chosen_labels_lst = [], [], [], []
            for i in range(len(batch)):
                chosen_input_ids, chosen_attention_mask, chosen_position_ids, chosen_labels = self.inner_wrap_batch(batch, i)
                chosen_input_ids_lst.append(chosen_input_ids)
                chosen_attention_mask_lst.append(chosen_attention_mask)
                chosen_position_ids_lst.append(chosen_position_ids)
                chosen_labels_lst.append(chosen_labels)
            
            wrap_batch["input_ids"] = torch.cat(chosen_input_ids_lst, dim=0)
            wrap_batch["attention_mask"] = torch.cat(chosen_attention_mask_lst, dim=0)
            wrap_batch["position_ids"] = torch.cat(chosen_position_ids_lst, dim=0)
            wrap_batch["labels"] = torch.cat(chosen_labels_lst, dim=0)
        
        return wrap_batch


class CustomTrainer(Trainer):
    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer, **kwargs):
        super().__init__(
            model=model, 
            args=args, 
            train_dataset=train_dataset, 
            eval_dataset=eval_dataset, 
            tokenizer=tokenizer,
            **kwargs,
        )    

    def move_to_device(self, batch, device):
        if isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = self.move_to_device(v, device)
        elif isinstance(batch, torch.Tensor):
            batch = batch.to(device)
        return batch

    def compute_loss(self, model, concatenated_batch, return_outputs=False):
        concatenated_batch = self.move_to_device(concatenated_batch, device=self.accelerator.device)
        outputs = model(**concatenated_batch, return_dict=True, use_cache=False)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss 


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    set_seed(script_args.seed)

    torch_dtype = torch.bfloat16
    config = AutoConfig.from_pretrained(script_args.model_name_or_path)
    if script_args.factor is not None:
        config.rope_scaling = {"type": script_args.rope_type, "factor": script_args.factor}
    if script_args.rope_theta is not None:
        config.rope_theta = script_args.rope_theta
    if script_args.max_position_embeddings is not None:
        config.max_position_embeddings = script_args.max_position_embeddings 
    
    base_model = LlamaForCausalLM.from_pretrained(
        script_args.model_name_or_path, config=config, 
        attn_implementation="flash_attention_2", torch_dtype=torch_dtype
    )

    if script_args.peft_model_path is not None:
        model = PeftModelForCausalLM.from_pretrained(base_model, script_args.peft_model_path, torch_dtype="auto")
    else:
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc_in", "fc_out", "wte"],
            bias="none",
            task_type="CAUSAL_LM",
            
        )
        model = get_peft_model(model=base_model, peft_config=peft_config, adapter_name="context_scaling")

    model.print_trainable_parameters()
    model.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print_c("load datasets", c="green")
    train_dataset = auto_read_data(os.path.join(script_args.dataset_path, "train.pkl"))  # just for debugging
    dev_dataset = auto_read_data(os.path.join(script_args.dataset_path, "valid.pkl"))[:400]

    print_c("4. initialize training arguments:", c="green")
    
    training_args = DPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy=script_args.evaluation_strategy,
        eval_steps=script_args.eval_steps,
        load_best_model_at_end=script_args.load_best_model_at_end,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        save_total_limit=script_args.save_total_limit,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs=dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant),
        seed=script_args.seed,
        deepspeed=script_args.deepspeed,
        max_length=script_args.max_length,
        max_prompt_length=script_args.max_prompt_length,
        max_target_length=script_args.max_length - script_args.max_prompt_length,
        gamma=script_args.gamma,
        # model_adapter_name="trainable",  # use this when multiple adapters are used
        # ref_adapter_name="reference", # use this when multiple adapters are used
    )

    # 5. initialize the SFT trainer
    trainer = CustomTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=CustomDataCollator()
    )


    # 6. train
    trainer.train()
    trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)