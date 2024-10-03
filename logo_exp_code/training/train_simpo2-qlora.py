import os
import torch
from datasets import load_from_disk
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel, PeftModelForCausalLM
from transformers import AutoTokenizer, HfArgumentParser, set_seed, AutoConfig,BitsAndBytesConfig
from dpo_trainer import DPOTrainer
from typing import Union
from modelzipper.tutils import *
import torch.nn.functional as F
from arguments import ScriptArguments, DPOConfig
from llama import LlamaForCausalLM
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from peft.tuners.lora import LoraLayer


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

        len_chosen, all_spe_pos = batch['chosen_answer']['input_ids'].size(0), batch["all_spe_pos"]
        concatenated_batch = self.move_to_device(batch, device=self.accelerator.device)
        chosen_input_ids = torch.concatenate([concatenated_batch["input_ids"], batch['chosen_answer']['input_ids']], dim=1) 
        prefix_rejected_input_ids = torch.concatenate([concatenated_batch["input_ids"], batch['prefix_rejected_answer']['input_ids']], dim=1)
        suffix_rejected_input_ids = torch.concatenate([concatenated_batch["input_ids"], batch['suffix_rejected_answer']['input_ids']], dim=1)
        chosen_attention_mask = torch.concatenate([concatenated_batch["attention_mask"], batch['chosen_answer']['attention_mask']], dim=1)
        prefix_attention_mask = torch.concatenate([concatenated_batch["attention_mask"], batch['prefix_rejected_answer']['attention_mask']], dim=1)
        suffix_attention_mask = torch.concatenate([concatenated_batch["attention_mask"], batch['suffix_rejected_answer']['attention_mask']], dim=1)
        chosen_position_ids = torch.concatenate([concatenated_batch["position_ids"], batch['chosen_answer']['position_ids']], dim=1)
        prefix_position_ids = torch.concatenate([concatenated_batch["position_ids"], batch['prefix_rejected_answer']['position_ids']], dim=1)
        suffix_position_ids = torch.concatenate([concatenated_batch["position_ids"], batch['suffix_rejected_answer']['position_ids']], dim=1)
        
        wrap_input_ids = torch.concatenate([chosen_input_ids, prefix_rejected_input_ids, suffix_rejected_input_ids], dim=0)
        wrap_attention_mask = torch.concatenate([chosen_attention_mask, prefix_attention_mask, suffix_attention_mask], dim=0)
        wrap_position_ids = torch.concatenate([chosen_position_ids, prefix_position_ids, suffix_position_ids], dim=0)

        import pdb; pdb.set_trace()
        all_logits = model(wrap_input_ids, attention_mask=wrap_attention_mask, position_ids=wrap_position_ids, use_cache=True, return_dict=True).logits
        import pdb; pdb.set_trace()
        all_logps = self.get_batch_logps(
            all_logits[:, -512:, :],
            torch.concat([batch['chosen_answer']['labels'], batch['prefix_rejected_answer']['labels'], batch['suffix_rejected_answer']['labels']], dim=0)[:, -512:],
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
        # sft_loss = 0
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


def get_accelerate_model(args, model_config):

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    if args.full_finetune: assert args.bits in [16, 32]

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = torch.bfloat16
    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=model_config,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    if not args.full_finetune:
        print(f'adding LoRA modules...')
        modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc_in", "fc_out", "wte"]
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model, tokenizer


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
    
    model, tokenizer = get_accelerate_model(script_args, model_config=config)
    """
    base_model = LlamaForCausalLM.from_pretrained(
        script_args.model_name_or_path, config=config, 
        attn_implementation="flash_attention_2", torch_dtype=torch_dtype,
        load_in_4bit = True, 
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ),
    )

    if not script_args.peft_model_path is not None:
        model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=script_args.gradient_checkpointing)

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
            init_lora_weights="pissa"
        )
        model = get_peft_model(model=base_model, peft_config=peft_config, adapter_name="context_scaling")

    model.print_trainable_parameters()
    
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    """
    
    print_c("2. load datasets", c="green")
    dataset = load_from_disk(script_args.dataset_path)
    train_test_split = dataset.train_test_split(test_size=20, shuffle=True)
    train_dataset = train_test_split['train']
    dev_dataset = train_test_split['test']
    load_from_cache_path = True

    print_c("3. initialize training arguments:", c="green")
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