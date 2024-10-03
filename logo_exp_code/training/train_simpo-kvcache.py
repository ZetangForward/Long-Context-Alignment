import os
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional
from datasets import Dataset, load_dataset, load_from_disk, DatasetDict
from peft import LoraConfig, PeftModel, get_peft_model, PeftModelForCausalLM
from transformers import (AutoModelForCausalLM, AutoTokenizer, 
                          HfArgumentParser, set_seed, AutoConfig)
from dpo_trainer import DPOTrainer
from typing import List, Tuple, Union, Literal, Dict
from modelzipper.tutils import *
import torch.nn.functional as F
from scipy.stats import qmc
import numpy as np
from arguments import ScriptArguments, DPOConfig
from llama import LlamaForCausalLM


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )
    
def auto_padding(t: torch.Tensor, length: int, filling_value=-100, return_attention_mask=False):
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


class CustomDataCollator:
    def __call__(self, batch):
        wrap_batch = {}
        candicate_pools = ["chosen_answer", "prefix_rejected_answer", "suffix_rejected_answer"]
        keys = ["input_ids", "attention_mask", "labels", "position_ids"]
        if len(batch) == 1:
            wrap_batch["all_spe_pos"] = batch[0]["all_spe_pos"]
            wrap_batch["input_ids"] = torch.tensor(batch[0]["input_ids"]).unsqueeze(0)
            wrap_batch["attention_mask"] = torch.tensor(batch[0]["attention_mask"]).unsqueeze(0)
            wrap_batch["position_ids"] = torch.tensor(batch[0]["position_ids"]).unsqueeze(0)
            for candicate_key in candicate_pools:
                wrap_batch[candicate_key] = {}
                for key in keys:
                    wrap_batch[candicate_key][key] = torch.tensor(batch[0][candicate_key][key]).unsqueeze(0)
        else:
            input_ids, attention_masks, position_ids = [], [], []
            candidate_pools = ["chosen_answer", "prefix_rejected_answer", "suffix_rejected_answer"]
            candidate_data = {candidate: {key: [] for key in keys} for candidate in candidate_pools}
            for sample in batch:
                input_ids.append(torch.tensor(sample["input_ids"]))
                attention_masks.append(torch.tensor(sample["attention_mask"]))
                position_ids.append(torch.tensor(sample["position_ids"]))
                for candidate_key in candidate_pools:
                    for key in keys:
                        candidate_data[candidate_key][key].append(torch.tensor(sample[candidate_key][key]))
            wrap_batch["input_ids"] = torch.stack(input_ids, dim=0)
            wrap_batch["attention_mask"] = torch.stack(attention_masks, dim=0)
            wrap_batch["position_ids"] = torch.stack(position_ids, dim=0)
            for candidate_key in candidate_pools:
                wrap_batch[candidate_key] = {}
                for key in keys:
                    wrap_batch[candidate_key][key] = torch.stack(candidate_data[candidate_key][key], dim=0)
            import pdb; pdb.set_trace()

        return wrap_batch


class SimPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        training_args = kwargs["args"]
        self.gamma = training_args.gamma
        self.max_position_embeddings = self.model.config.max_position_embeddings
        self.max_chunk_size = training_args.max_chunk_size
        self.max_qa_size = training_args.max_qa_size

    def simpo_loss(self, policy_chosen_logps, policy_rejected_logps):
        """Compute the SimPO loss for a batch of policy model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the SimPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        gamma_logratios = self.gamma / self.beta 
        pi_logratios = pi_logratios.to(self.accelerator.device)
        logits = pi_logratios - gamma_logratios
        if self.loss_type == "sigmoid": 
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']"
            )
        chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
        return losses, chosen_rewards, rejected_rewards
    

    def concatenated_forward(self, model, batch): # one chosen, two rejected
        len_chosen = batch['chosen_answer']['input_ids'].size(0)
        concatenated_batch = self.concatenated_inputs(batch, device=self.accelerator.device) 
        import pdb; pdb.set_trace()
        hidden_states = model(concatenated_batch["input_ids"], attention_mask=concatenated_batch["attention_mask"], position_ids=concatenated_batch["position_ids"], use_cache=True)
        kv_cache = hidden_states.past_key_values

        import pdb; pdb.set_trace()

        chosen_answer = batch['chosen_answer']
        prefix_rejected_answer = batch['prefix_rejected_answer']
        suffix_rejected_answer = batch['suffix_rejected_answer']
        chosen_logits = model(chosen_answer['input_ids'], attention_mask=chosen_answer['attention_mask'], past_key_values=kv_cache, use_cache=True).logits
        prefix_chosen_logits = model(prefix_rejected_answer['input_ids'], attention_mask=prefix_rejected_answer['attention_mask'], past_key_values=kv_cache, use_cache=True).logits
        suffix_chosen_logits = model(suffix_rejected_answer['input_ids'], attention_mask=suffix_rejected_answer['attention_mask'], past_key_values=kv_cache, use_cache=True).logits

        chosen_answer_labels = chosen_answer['labels'][:, concatenated_batch["input_ids"].size(1):]
        prefix_rejected_answer_labels = prefix_rejected_answer['labels'][:, concatenated_batch["input_ids"].size(1):]
        suffix_rejected_answer_labels = suffix_rejected_answer['labels'][:, concatenated_batch["input_ids"].size(1):]

        chosen_logits = torch.concat([hidden_states.logits[:, -1, :][:, None, :], chosen_logits[:, :-1, :]], dim=1).contiguous()
        prefix_chosen_logits = torch.concat([hidden_states.logits[:, -1, :][:, None, :], prefix_chosen_logits[:, :-1, :]], dim=1).contiguous()
        suffix_chosen_logits = torch.concat([hidden_states.logits[:, -1, :][:, None, :], suffix_chosen_logits[:, :-1, :]], dim=1).contiguous()

        # normal CE Loss
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_logits = chosen_logits.view(-1, self.model.config.vocab_size)
        shift_labels = chosen_answer_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        sft_loss = loss_fct(shift_logits, shift_labels)

        all_logps = self.get_batch_logps(
            torch.concat([chosen_logits, prefix_chosen_logits, suffix_chosen_logits], dim=0),
            torch.concat([chosen_answer_labels, prefix_rejected_answer_labels, suffix_rejected_answer_labels], dim=0),
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        # all_logps = self.get_batch_logps(
        #     all_logits[:, -1024:, :],
        #     torch.concat([batch['chosen_answer']['labels'], batch['prefix_rejected_answer']['labels'], batch['suffix_rejected_answer']['labels']], dim=0)[:, -1024:],
        #     average_log_prob=True,
        #     is_encoder_decoder=self.is_encoder_decoder,
        #     label_pad_token_id=self.label_pad_token_id,
        # )

        chosen_logps = all_logps[:len_chosen]
        prefix_chosen_logps = all_logps[len_chosen:len_chosen*2]
        suffix_chosen_logps = all_logps[len_chosen*2:]
        # chosen_logits = all_logits[:len_chosen]
        # prefix_chosen_logits = all_logits[:len_chosen]
        # suffix_chosen_logits = all_logits[:len_chosen]

        return (chosen_logps, 
                prefix_chosen_logps, 
                suffix_chosen_logps, 
                chosen_logits, 
                prefix_chosen_logits, 
                suffix_chosen_logits,
                sft_loss)

    def get_batch_loss_metrics(self, model, batch, train_eval):
        """Compute the SimPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        policy_chosen_logps, policy_rejected_logps1, policy_rejected_logps2, policy_chosen_logits, policy_rejected_logits1, policy_rejected_logits2, sft_loss = self.concatenated_forward(model, batch)
        
        losses, chosen_rewards, rejected_rewards = self.simpo_loss(policy_chosen_logps, (policy_rejected_logps1 + policy_rejected_logps2) / 2)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
    
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}sft_loss"] = sft_loss.cpu()
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected1"] = policy_rejected_logps1.detach().mean().cpu()
        metrics[f"{prefix}logps/rejected2"] = policy_rejected_logps2.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected1"] = policy_rejected_logits1.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected2"] = policy_rejected_logits2.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return losses.mean() + 0.5 * sft_loss, metrics
    
    def move_to_device(self, batch, device):
        if isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = self.move_to_device(v, device)
        elif isinstance(batch, torch.Tensor):
            batch = batch.to(device)
        return batch

    def concatenated_inputs(self, batch, device):
        batch = self.move_to_device(batch, device)
        return batch


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    set_seed(script_args.seed)

    print_c("load models", c="green")
    torch_dtype = torch.bfloat16
    config = AutoConfig.from_pretrained(script_args.model_name_or_path)
    if script_args.factor is not None:
        config.rope_scaling = {"type": script_args.rope_type, "factor": script_args.factor}
    if script_args.rope_theta is not None:
        config.rope_theta = script_args.rope_theta
    if script_args.max_position_embeddings is not None:
        config.max_position_embeddings = script_args.max_position_embeddings 

    base_model = LlamaForCausalLM.from_pretrained(
        script_args.model_name_or_path, 
        config=config, 
        torch_dtype=torch_dtype,
        # device_map="balanced_low_0", 
    )

    if script_args.peft_model_path is not None:
        model = PeftModelForCausalLM.from_pretrained(base_model, script_args.peft_model_path, torch_dtype="auto")
    else:
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model=base_model, peft_config=peft_config, adapter_name="context_scaling")

    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print_c("load datasets", c="green")
    dataset = load_from_disk(script_args.dataset_path)
    train_test_split = dataset.train_test_split(test_size=20, shuffle=True)
    train_dataset = train_test_split['train']
    dev_dataset = train_test_split['test']
    load_from_cache_path = True

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

    # 5. initialize the DPO trainer
    dpo_trainer = SimPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,  # just for debug
        tokenizer=tokenizer,
        data_collator=CustomDataCollator(),
        # num_train_epochs=script_args.num_train_epochs,
        dataset_num_proc=script_args.dataset_num_proc,
        # peft_config=peft_config,
        force_use_ref_model=False,
        load_from_cache_path = load_from_cache_path,
        # cache_path = cache_path,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)