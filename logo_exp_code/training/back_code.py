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


class CustomDataCollator:

    def inner_wrap_batch(self, batch_data, batch_id):
        input_ids = torch.tensor(batch_data[batch_id]["input_ids"])
        attention_mask = torch.tensor(batch_data[batch_id]["attention_mask"])
        position_ids = torch.tensor(batch_data[batch_id]["position_ids"])
        chosen_answer_input_ids = batch_data[batch_id]["chosen_answer"]["input_ids"]
        prefix_rejected_answer_input_ids = batch_data[batch_id]["prefix_rejected_answer"]["input_ids"]
        suffix_rejected_answer_input_ids = batch_data[batch_id]["suffix_rejected_answer"]["input_ids"]
        chosen_answer_attention_mask = batch_data[batch_id]["chosen_answer"]["attention_mask"]
        prefix_rejected_answer_attention_mask = batch_data[batch_id]["prefix_rejected_answer"]["attention_mask"]
        suffix_rejected_answer_attention_mask = batch_data[batch_id]["suffix_rejected_answer"]["attention_mask"]
        chosen_answer_position_ids = batch_data[batch_id]["chosen_answer"]["position_ids"]
        prefix_rejected_answer_position_ids = batch_data[batch_id]["prefix_rejected_answer"]["position_ids"]
        suffix_rejected_answer_position_ids = batch_data[batch_id]["suffix_rejected_answer"]["position_ids"]

        chosen_input_ids = torch.concatenate([input_ids, chosen_answer_input_ids], dim=0).unsqueeze(0)
        chosen_attention_mask = torch.concatenate([attention_mask, chosen_answer_attention_mask], dim=0).unsqueeze(0)
        chosen_position_ids = torch.concatenate([position_ids, chosen_answer_position_ids], dim=0).unsqueeze(0)
        chosen_labels = torch.tensor(batch_data[batch_id]["chosen_answer"]["labels"]).unsqueeze(0)
        
        prefix_rejected_input_ids = torch.concatenate([input_ids, prefix_rejected_answer_input_ids], dim=0).unsqueeze(0)
        prefix_rejected_attention_mask = torch.concatenate([attention_mask, prefix_rejected_answer_attention_mask], dim=0).unsqueeze(0)
        prefix_rejected_position_ids = torch.concatenate([position_ids, prefix_rejected_answer_position_ids], dim=0).unsqueeze(0)
        prefix_rejected_labels = torch.tensor(batch_data[batch_id]["prefix_rejected_answer"]["labels"]).unsqueeze(0)

        suffix_rejected_input_ids = torch.concatenate([input_ids, suffix_rejected_answer_input_ids], dim=0).unsqueeze(0)
        suffix_rejected_attention_mask = torch.concatenate([attention_mask, suffix_rejected_answer_attention_mask], dim=0).unsqueeze(0)
        suffix_rejected_position_ids = torch.concatenate([position_ids, suffix_rejected_answer_position_ids], dim=0).unsqueeze(0)
        suffix_rejected_labels = torch.tensor(batch_data[batch_id]["suffix_rejected_answer"]["labels"]).unsqueeze(0)

        return chosen_input_ids, chosen_attention_mask, chosen_position_ids, chosen_labels, prefix_rejected_input_ids, prefix_rejected_attention_mask, prefix_rejected_position_ids, prefix_rejected_labels, suffix_rejected_input_ids, suffix_rejected_attention_mask, suffix_rejected_position_ids, suffix_rejected_labels

    def __call__(self, batch):
        wrap_batch = {}

        if len(batch) == 1:
            chosen_input_ids, chosen_attention_mask, chosen_position_ids, chosen_labels, prefix_rejected_input_ids, prefix_rejected_attention_mask, prefix_rejected_position_ids, prefix_rejected_labels, suffix_rejected_input_ids, suffix_rejected_attention_mask, suffix_rejected_position_ids, suffix_rejected_labels = self.inner_wrap_batch(batch, 0)

            wrap_batch["input_ids"] = chosen_input_ids
            wrap_batch["attention_mask"] = chosen_attention_mask
            wrap_batch["position_ids"] = chosen_position_ids
            wrap_batch["chosen_labels"] = chosen_labels
            wrap_batch["prefix_rejected_input_ids"] = prefix_rejected_input_ids
            wrap_batch["prefix_rejected_attention_mask"] = prefix_rejected_attention_mask
            wrap_batch["prefix_rejected_position_ids"] = prefix_rejected_position_ids
            wrap_batch["prefix_rejected_labels"] = prefix_rejected_labels
            wrap_batch["suffix_rejected_input_ids"] = suffix_rejected_input_ids
            wrap_batch["suffix_rejected_attention_mask"] = suffix_rejected_attention_mask
            wrap_batch["suffix_rejected_position_ids"] = suffix_rejected_position_ids
            wrap_batch["suffix_rejected_labels"] = suffix_rejected_labels
            
        else:
            chosen_input_ids_lst, chosen_attention_mask_lst, chosen_position_ids_lst, chosen_labels_lst, prefix_rejected_input_ids_lst, prefix_rejected_attention_mask_lst, prefix_rejected_position_ids_lst, prefix_rejected_labels_lst, suffix_rejected_input_ids_lst, suffix_rejected_attention_mask_lst, suffix_rejected_position_ids_lst, suffix_rejected_labels_lst = [], [], [], [], [], [], [], [], [], [], [], []
            for i in range(len(batch)):
                chosen_input_ids, chosen_attention_mask, chosen_position_ids, chosen_labels, prefix_rejected_input_ids, prefix_rejected_attention_mask, prefix_rejected_position_ids, prefix_rejected_labels, suffix_rejected_input_ids, suffix_rejected_attention_mask, suffix_rejected_position_ids, suffix_rejected_labels = self.inner_wrap_batch(batch, i)
                chosen_input_ids_lst.append(chosen_input_ids)
                chosen_attention_mask_lst.append(chosen_attention_mask)
                chosen_position_ids_lst.append(chosen_position_ids)
                chosen_labels_lst.append(chosen_labels)
                prefix_rejected_input_ids_lst.append(prefix_rejected_input_ids)
                prefix_rejected_attention_mask_lst.append(prefix_rejected_attention_mask)
                prefix_rejected_position_ids_lst.append(prefix_rejected_position_ids)
                prefix_rejected_labels_lst.append(prefix_rejected_labels)
                suffix_rejected_input_ids_lst.append(suffix_rejected_input_ids)
                suffix_rejected_attention_mask_lst.append(suffix_rejected_attention_mask)
                suffix_rejected_position_ids_lst.append(suffix_rejected_position_ids)
                suffix_rejected_labels_lst.append(suffix_rejected_labels)
            
            wrap_batch["input_ids"] = torch.cat(chosen_input_ids_lst, dim=0)
            wrap_batch["attention_mask"] = torch.cat(chosen_attention_mask_lst, dim=0)
            wrap_batch["position_ids"] = torch.cat(chosen_position_ids_lst, dim=0)
            wrap_batch["chosen_labels"] = torch.cat(chosen_labels_lst, dim=0)
            wrap_batch["prefix_rejected_input_ids"] = torch.cat(prefix_rejected_input_ids_lst, dim=0)
            wrap_batch["prefix_rejected_attention_mask"] = torch.cat(prefix_rejected_attention_mask_lst, dim=0)
            wrap_batch["prefix_rejected_position_ids"] = torch.cat(prefix_rejected_position_ids_lst, dim=0)
            wrap_batch["prefix_rejected_labels"] = torch.cat(prefix_rejected_labels_lst, dim=0)
            wrap_batch["suffix_rejected_input_ids"] = torch.cat(suffix_rejected_input_ids_lst, dim=0)
            wrap_batch["suffix_rejected_attention_mask"] = torch.cat(suffix_rejected_attention_mask_lst, dim=0)
            wrap_batch["suffix_rejected_position_ids"] = torch.cat(suffix_rejected_position_ids_lst, dim=0)
            wrap_batch["suffix_rejected_labels"] = torch.cat(suffix_rejected_labels_lst, dim=0)
        
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
        len_chosen = batch['input_ids'].size(0)
        concatenated_batch = self.move_to_device(batch, device=self.accelerator.device)

        input_ids = torch.concatenate([concatenated_batch["input_ids"], concatenated_batch["prefix_rejected_input_ids"], concatenated_batch["suffix_rejected_input_ids"]], dim=0)
        attention_mask = torch.concatenate([concatenated_batch["attention_mask"], concatenated_batch["prefix_rejected_attention_mask"], concatenated_batch["suffix_rejected_attention_mask"]], dim=0)
        position_ids = torch.concatenate([concatenated_batch["position_ids"], concatenated_batch["prefix_rejected_position_ids"], concatenated_batch["suffix_rejected_position_ids"]], dim=0)
        labels = torch.cat([concatenated_batch["chosen_labels"], concatenated_batch["prefix_rejected_labels"], concatenated_batch["suffix_rejected_labels"]], dim=0)    
        all_logits = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False, return_dict=True).logits
        all_logps = self.get_batch_logps(
            all_logits[:, -256:, :],
            labels[:, -256:],
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        prefix_chosen_logps = all_logps[len_chosen:len_chosen*2]
        suffix_chosen_logps = all_logps[len_chosen*2:]
        chosen_logits = all_logits[:len_chosen]
        prefix_chosen_logits = all_logits[:len_chosen]
        suffix_chosen_logits = all_logits[:len_chosen]
        chosen_logits = all_logits[:len_chosen]
        prefix_chosen_logits = all_logits[:len_chosen]
        suffix_chosen_logits = all_logits[:len_chosen]

        return (chosen_logps, 
                prefix_chosen_logps, 
                suffix_chosen_logps, 
                chosen_logits, 
                prefix_chosen_logits, 
                suffix_chosen_logits, )

    def get_batch_loss_metrics(self, model, batch, train_eval):
        """Compute the SimPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        policy_chosen_logps, policy_rejected_logps1, policy_rejected_logps2, policy_chosen_logits, policy_rejected_logits1, policy_rejected_logits2 = self.concatenated_forward(model, batch)
        
        losses, chosen_rewards, rejected_rewards = self.simpo_loss(policy_chosen_logps, (policy_rejected_logps1 + policy_rejected_logps2) / 2)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        prefix = "eval_" if train_eval == "eval" else ""
        # metrics[f"{prefix}sft_loss"] = sft_loss.cpu()
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

        return losses.mean(), metrics
    
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

    print_c("1. load models", c="green")
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

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print_c("load datasets", c="green")
    train_dataset = auto_read_data(os.path.join(script_args.dataset_path, "train.pkl"))  # just for debugging
    dev_dataset = auto_read_data(os.path.join(script_args.dataset_path, "valid.pkl"))[:400]
    # train_test_split = dataset.train_test_split(test_size=20, shuffle=True)
    # train_dataset = train_test_split['train']
    # dev_dataset = train_test_split['test']
    load_from_cache_path = True

    # cache_path = os.path.join(
    #     os.path.dirname(script_args.dataset_path), 
    #     os.path.basename(script_args.dataset_path).split('.')[0] + '_cache'
    # )
    # print("load datasets")
    # if os.path.exists(cache_path) and os.path.isdir(cache_path):
    #     all_data = load_from_disk(cache_path)
    #     train_dataset, eval_dataset = all_data['train'], all_data['valid']
    #     load_from_cache_path = True
    # else:
    #     all_data = load_from_disk(script_args.dataset_path)
    #     train_dataset, eval_dataset = all_data['train'], all_data['valid']  # set for debug

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