import os
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional
from datasets import Dataset, load_dataset, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoTokenizer, 
    HfArgumentParser, 
    set_seed, 
    AutoConfig, 
)
from dpo_trainer import DPOTrainer
from llama import SparseIDLlamaForCausalLM
from contextlib import contextmanager, nullcontext
import torch.nn as nn
from typing import List, Tuple, Union, Literal, Dict
from modelzipper.tutils import *
from arguments import *


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


class CustomDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Make sure to move the loss to the device the original accumulating loss is at back in the `Trainer` class:
        loss = loss.to(self.args.device)
        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss


    def get_batch_loss_metrics(self, model, batch, train_eval):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, policy_rejected_logits \
            = self.concatenated_forward(model, batch)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch, use_adapter=False)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch, use_adapter=False)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        # print(chosen_rewards.mean(), rejected_rewards.mean(), reward_accuracies.mean())
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()
        return losses.mean(), metrics


    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], use_adapter=True,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        concatenated_batch = self.concatenated_inputs(batch, is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id, padding_value=self.padding_value, device=self.accelerator.device)
        len_chosen = batch["chosen_labels"].shape[0]
        # print_c(f"current sequence length {concatenated_batch['concatenated_input_ids'].shape[-1]}", 'magenta')
        
        model_kwargs = ({"labels": concatenated_batch["concatenated_labels"], "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None)} if self.is_encoder_decoder else {})
        chosen_input_ids, reject_input_ids = concatenated_batch["concatenated_input_ids"].split(1, dim=0)
        chosen_attention_mask, reject_attention_mask = concatenated_batch["concatenated_attention_mask"].split(1, dim=0)

        if not use_adapter: # use the model directly (ref_model)
            with model.disable_adapter():  # base model (without lora)
                """  
                ## original input code (forward batch once, with batch size = 2)
                    all_logits = model(
                        concatenated_batch["concatenated_input_ids"],
                        attention_mask=concatenated_batch["concatenated_attention_mask"],
                        use_cache=False,
                        **model_kwargs,
                    ).logits    
                """
                chosen_all_logits = model(chosen_input_ids, attention_mask=chosen_attention_mask, use_cache=False, **model_kwargs).logits
                reject_all_logits = model(reject_input_ids, attention_mask=reject_attention_mask, use_cache=False, **model_kwargs).logits
        else:  # utilize the policy model
            """  
            ## original input code (forward batch once, with batch size = 2)
                all_logits = model(
                    concatenated_batch["concatenated_input_ids"],
                    attention_mask=concatenated_batch["concatenated_attention_mask"],
                    use_cache=False,
                    **model_kwargs,
                ).logits    
            """
            chosen_all_logits = model(chosen_input_ids, attention_mask=chosen_attention_mask, use_cache=False, **model_kwargs).logits
            reject_all_logits = model(reject_input_ids, attention_mask=reject_attention_mask, use_cache=False, **model_kwargs).logits
            
        # merge chosen and reject logits
        all_logits = torch.concatenate([chosen_all_logits, reject_all_logits], dim=0)
        
        all_logps = self.get_batch_logps(all_logits, concatenated_batch["concatenated_labels"], 
                                         average_log_prob=self.loss_type == "ipo", 
                                         is_encoder_decoder=self.is_encoder_decoder, 
                                         label_pad_token_id=self.label_pad_token_id)
      
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0
        
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def tokenize_row(self, feature, model = None):
        batch = {}
        prompt, chosen, rejected = feature["prompt"], feature["chosen"], feature["rejected"]
        chosen_span_id, rejected_span_id = feature['chosen_span_id'], feature['rejected_span_id']

        if not self.is_encoder_decoder:

            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)}")
            chosen_tokens = self.build_tokenized_answer(prompt, chosen)

            if not isinstance(rejected, str):
                raise ValueError(f"rejected should be an str but got {type(rejected)}")
            rejected_tokens = self.build_tokenized_answer(prompt, rejected)

            # Last prompt token might get merged by tokenizer and
            # it should not be included for generation if that happens
            prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

            chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
            rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
            prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

            for k, v in prompt_tokens.items(): # Make sure prompt is not too long
                prompt_tokens[k] = v[:prompt_len_input_ids]

            # Make sure prompts only have one different token at most an
            # and length only differs by 1 at most
            num_diff_tokens = sum(
                [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
            )
            num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
            
            if num_diff_tokens > 1 or num_diff_len > 1:
                raise ValueError(
                    "Chosen and rejected prompt_input_ids might only differ on the "
                    "last token due to tokenizer merge ops."
                )

            # add BOS token to head of prompt. Avoid adding if it's already there
            bos_token_id = self.tokenizer.bos_token_id
            if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
                prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
                prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
            if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
                chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
                chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
            if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
                rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
                rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

            # add EOS token to end of answer. Avoid adding if it's already there
            eos_token_id = self.tokenizer.eos_token_id
            if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
                chosen_tokens["input_ids"].append(eos_token_id)
                chosen_tokens["attention_mask"].append(1)
            if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
                rejected_tokens["input_ids"].append(eos_token_id)
                rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))
            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")
            
            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            # Create labels
            chosen_sequence_tokens = {k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]}
            rejected_sequence_tokens = {k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]}
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]  # prompt + chosen answer
            chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [self.label_pad_token_id] * len(chosen_tokens["prompt_input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:] # prompt + rejected answer
            rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [self.label_pad_token_id] * len(rejected_tokens["prompt_input_ids"])
            
            for k, toks in {"chosen_": chosen_sequence_tokens, "rejected_": rejected_sequence_tokens, "": prompt_tokens}.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens
                    if f"{k}{type_key}" == "chosen_attention_mask":
                        batch["sequence_length"] = len(tokens)

        else:
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.tokenizer(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]
            batch["chosen_span_id"] = chosen_span_id
            batch["rejected_span_id"] = rejected_span_id
            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["rejected_labels"])
                )
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["chosen_labels"])
                )

        return batch

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}
        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key], 
                        pad_to_length(batch[k], max_length, pad_value=pad_value)), 
                        dim=0
                    ).to(device=device)
        
        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )
        
        return concatenated_batch


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)

    # 1. load a pretrained model
    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    config = AutoConfig.from_pretrained(script_args.model_name_or_path)
    config.rope_scaling = {"type": script_args.rope_type, "factor": script_args.factor}
    config.rope_theta = script_args.rope_theta
    config.max_position_embeddings = config.max_position_embeddings * script_args.rope_theta
    
    base_model = SparseIDLlamaForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        config=config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    base_model.config.use_cache = False
    
    # set the lora config
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model=base_model, peft_config=peft_config, adapter_name="trainable")
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Load the training datasset
    cache_path = os.path.join(os.path.dirname(script_args.dataset_path), os.path.basename(script_args.dataset_path).split('.')[0] + '_cache')

    load_from_cache_path = False
    if os.path.exists(cache_path) and os.path.isdir(cache_path):
        all_data = load_from_disk(cache_path)
        train_dataset, eval_dataset = all_data['train'], all_data['valid']
        load_from_cache_path = True
    else:
        all_data = load_from_disk(script_args.dataset_path)
        train_dataset, eval_dataset = all_data['train'], all_data['valid']  # set for debug
   
    # 4. initialize training arguments:
    training_args = DPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        eval_strategy=script_args.eval_strategy,
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
        # run_name=script_args.run_name,
        gradient_checkpointing_kwargs=dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant),
        seed=script_args.seed,
        deepspeed=script_args.deepspeed,
        max_length=script_args.max_length,
        max_prompt_length=script_args.max_prompt_length,
        max_target_length=script_args.max_length - script_args.max_prompt_length,
        # model_adapter_name="trainable",
        # ref_adapter_name="reference",   
    )

    # 5. initialize the DPO trainer
    dpo_trainer = CustomDPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # num_train_epochs=script_args.num_train_epochs,
        dataset_num_proc=script_args.dataset_num_proc,
        # peft_config=peft_config,
        force_use_ref_model=False,
        load_from_cache_path = load_from_cache_path,
        cache_path = cache_path,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)