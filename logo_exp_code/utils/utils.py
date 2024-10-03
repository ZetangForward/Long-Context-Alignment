import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoTokenizer,
    TrainingArguments,
    AutoConfig
)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def create_and_prepare_model(model_id: str, training_args: TrainingArguments, model_args):

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    if model_args.rope_type is not None:
        config.rope_scaling = {"type": model_args.rope_type, "factor": model_args.factor}
    if model_args.rope_theta is not None:
        config.rope_theta = model_args.rope_theta

    model = AutoModelForCausalLM.from_pretrained(
        model_id, config=config, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    print("model loaded")

    if training_args.low_rank_training:
        # initialize peft model
        print("initializing peft model")
        model.enable_input_require_grads()  # enable input requires grad
        if model_args.peft_model_path is not None:
            model = PeftModelForCausalLM.from_pretrained(model, model_args.peft_model_path)
        else:
            if model_args.model_type == "gpt-neox":
                targets = ["query_key_value", "dense"]
            else:
                targets=["q_proj", "k_proj", "v_proj", "o_proj"]
            peft_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=targets,
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)

    # enable gradient checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # pre-process the model by upcasting the layer norms in float 32 for
    # Adapted from https://github.com/tmm1/axolotl/blob/2eda9e02a9d15a7a3f92b41f257d9844d72fc220/src/axolotl/utils/models.py#L338
    print("pre-processing model for peft")
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.bfloat16)
        if any(x in name for x in ["lm_head", "embed_tokens", "wte", "wpe"]):
            if hasattr(module, "weight"):
                module = module.to(torch.bfloat16)

    # logger.info parameters
    model.print_trainable_parameters()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer