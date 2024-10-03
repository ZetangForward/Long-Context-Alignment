import os
import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM
from transformers import (AutoTokenizer, HfArgumentParser, set_seed, AutoConfig)
from simpo_trainer import SimPOTrainer
from typing import Union
from modelzipper.tutils import *
import torch.nn.functional as F
from arguments import ScriptArguments, DPOConfig, SimPOConfig
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


class MultiObj_Simpo_Trainer(SimPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def simpo_loss(self, policy_chosen_logps, policy_rejected_logps):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        pi_logratios = pi_logratios.to(self.accelerator.device)
        logits = pi_logratios - self.gamma_beta_ratio
        
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
        prefix_chosen_logits = all_logits[len_chosen:len_chosen*2]
        suffix_chosen_logits = all_logits[len_chosen*2:]

        return (chosen_logps, 
                prefix_chosen_logps, 
                suffix_chosen_logps, 
                chosen_logits, 
                prefix_chosen_logits, 
                suffix_chosen_logits, )

    def get_batch_loss_metrics(self, model, batch, train_eval):
        """Compute the SimPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""
        
        policy_chosen_logps, policy_rejected_logps1, policy_rejected_logps2, policy_chosen_logits, policy_rejected_logits1, policy_rejected_logits2 = self.concatenated_forward(model, batch)
        
        losses, chosen_rewards, rejected_rewards = self.simpo_loss(policy_chosen_logps, (policy_rejected_logps1 + policy_rejected_logps2) / 2)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        loss = losses.mean()

        if self.sft_weight > 0.0:
            loss_func = nn.CrossEntropyLoss() # method 2
            sft_loss = loss_func(policy_chosen_logits.view(-1, policy_chosen_logits.shape[-1]), batch["chosen_labels"].view(-1))
            loss = self.sft_weight * sft_loss + loss
            metrics[f"{prefix}sft_loss"] = sft_loss.detach().cpu()
    
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

        return loss, metrics
    


if __name__ == "__main__":
    parser = HfArgumentParser(SimPOConfig)
    training_args = parser.parse_args_into_dataclasses()[0]
    set_seed(training_args.seed)

    print_c("1. load models", c="green")
    torch_dtype = torch.bfloat16
    config = AutoConfig.from_pretrained(training_args.model_name_or_path)
    if training_args.rope_type is not None:
        config.rope_scaling = {"type": training_args.rope_type, "factor": training_args.factor}
    if training_args.rope_theta is not None:
        config.rope_theta = training_args.rope_theta
    if training_args.max_position_embeddings is not None:
        config.max_position_embeddings = training_args.max_position_embeddings 
    
    config.use_logn_scaling = False
    
    model = LlamaForCausalLM.from_pretrained(
        training_args.model_name_or_path, config=config, 
        attn_implementation="flash_attention_2", torch_dtype=torch_dtype
    )

    if training_args.peft_model_path is not None:
        model = PeftModelForCausalLM.from_pretrained(model, training_args.peft_model_path, torch_dtype="auto")
    elif not training_args.full_finetune:
        peft_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc_in", "fc_out", "wte"],
            bias="none",
            task_type="CAUSAL_LM",
            
        )
        model = get_peft_model(model=model, peft_config=peft_config, adapter_name="context_scaling")
        model.print_trainable_parameters()
    else:
        log_c("begin full parameters tuning ...")

    tokenizer = AutoTokenizer.from_pretrained(training_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print_c("load datasets", c="green")
    train_dataset = auto_read_data(os.path.join(training_args.dataset_path, "train.pkl"))  # just for debugging
    dev_dataset = auto_read_data(os.path.join(training_args.dataset_path, "valid.pkl"))
    
    random.shuffle(train_dataset)
    random.shuffle(dev_dataset)

    # train_test_split = dataset.train_test_split(test_size=20, shuffle=True)
    # train_dataset = train_test_split['train']
    # dev_dataset = train_test_split['test']
    load_from_cache_path = True

    # cache_path = os.path.join(
    #     os.path.dirname(training_args.dataset_path), 
    #     os.path.basename(training_args.dataset_path).split('.')[0] + '_cache'
    # )
    # print("load datasets")
    # if os.path.exists(cache_path) and os.path.isdir(cache_path):
    #     all_data = load_from_disk(cache_path)
    #     train_dataset, eval_dataset = all_data['train'], all_data['valid']
    #     load_from_cache_path = True
    # else:
    #     all_data = load_from_disk(training_args.dataset_path)
    #     train_dataset, eval_dataset = all_data['train'], all_data['valid']  # set for debug

    print_c("4. initialize training arguments:", c="green")

    # 5. initialize the DPO trainer
    dpo_trainer = MultiObj_Simpo_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,  # just for debug
        tokenizer=tokenizer,
        data_collator=CustomDataCollator(),
        load_from_cache_path = load_from_cache_path,
        cache_path = None,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(training_args.output_dir)

    # 7. save
    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)