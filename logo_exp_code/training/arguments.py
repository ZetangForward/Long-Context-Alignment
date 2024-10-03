import os
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional
from datasets import Dataset, load_dataset, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import TrainingArguments
import transformers
from typing import List, Tuple, Union, Literal, Dict
from modelzipper.tutils import *


@dataclass
class SimPOConfig(TrainingArguments):
    r"""
    SimPOConfig collects all training arguments related to the [`SimPOTrainer`] class.

    Using [`HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_target_length (`int`, defaults to `None`):
            The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        beta (`float`, defaults to 2.0):
            The beta factor in SimPO loss.
        gamma_beta_ratio (`float`, defaults to 0.25):
            The ratio between the target reward margin (gamma) and beta in SimPO loss.
        sft_weight (`float`, defaults to 0.0):
            SFT loss weight added to the SimPO loss (0.0 is not using SFT).
        label_smoothing (`float`, defaults to 0):
            The label smoothing factor. This argument is required if you want to use the default data collator.
        loss_type (`str`, defaults to `sigmoid`):
            The type of loss to use. This argument is required if you want to use the default data collator.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `None`):
            The padding value if it is different to the tokenizer's pad_token_id.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        generate_during_eval (`bool`, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model`.
        model_init_kwargs (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string
        dataset_num_proc (`Optional[int]`, *optional*):
            The number of workers to use to tokenize the data. Defaults to None.
    """
    
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_completion_length: Optional[int] = None
    max_target_length: Optional[int] = None

    beta: float = 2.0
    gamma_beta_ratio: float = 0.25
    sft_weight: float = 0.0
    label_smoothing: float = 0
    loss_type: Literal["sigmoid", "hinge"] = "sigmoid"
    disable_dropout: bool = True

    label_pad_token_id: int = -100
    padding_value: int = None
    truncation_mode: str = "keep_end"
    generate_during_eval: bool = False
    is_encoder_decoder: Optional[bool] = None

    model_init_kwargs: Optional[Dict] = None
    dataset_num_proc: Optional[int] = None

    model_name_or_path: str = None
    max_position_embeddings: Optional[int] = None
    model_dtype: Optional[str] = "bfloat16"

    full_finetune: bool = False
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    dataset_path: str = None
    rope_type: Optional[str] = None
    rope_factor: Optional[float] = None
    rope_theta: Optional[float] = None
    peft_model_path: Optional[str] = None


@dataclass
class DPOConfig(TrainingArguments):
    beta: float = 0.1
    label_smoothing: float = 0
    loss_type: Literal[
        "sigmoid", "hinge", "ipo", "kto_pair", "bco_pair", "sppo_hard", "nca_pair", "robust"
    ] = "sigmoid"
    label_pad_token_id: int = -100
    padding_value: int = 0
    truncation_mode: str = "keep_end"
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_target_length: Optional[int] = None
    is_encoder_decoder: Optional[bool] = None
    disable_dropout: bool = True
    generate_during_eval: bool = False
    precompute_ref_log_probs: bool = False
    dataset_num_proc: Optional[int] = None
    model_init_kwargs: Optional[Dict] = None
    ref_model_init_kwargs: Optional[Dict] = None
    model_adapter_name: Optional[str] = None
    ref_adapter_name: Optional[str] = None
    reference_free: bool = False
    force_use_ref_model: bool = False
    sync_ref_model: bool = False
    ref_model_mixup_alpha: float = 0.9
    ref_model_sync_steps: int = 64
    gamma: float = 0.5
    max_chunk_size = 256
    max_qa_size = 512

    

# Define and parse arguments.
@dataclass
class ScriptArguments(transformers.TrainingArguments):
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    # training parameters
    model_name_or_path: Optional[str] = field(
        default="../sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )
    peft_model_path: Optional[str] = field(default=None, metadata={"help": "peft model path"})
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=8192, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})
    
    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="float16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )
    dataset_num_proc: Optional[int] = field(default=48, metadata={"help": "the number of dataset processes"})
    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": 'Path to the dataset.'}
    )
    
    run_name: Optional[str] = field(
        default=None,
        metadata={"help": 'Experiment name'}
    )

    rope_type: Optional[str] = field(
        default=None, metadata={"help": "RoPE Type"}
    )

    factor: Optional[int] = field(
        default=None, metadata={"help": "Context Scaling Ratio"}
    )

    rope_theta: Optional[float] = field(
        default=100000.0, metadata={"help": "Base of RoPE"}
    )
    
    gamma: Optional[float] = field(
        default=0.5, metadata={"help": "Base of RoPE"}
    )

    max_chunk_size: Optional[int] = field(
        default=256, metadata={"help": "max chunk size"}
    )

    max_qa_size: Optional[int] = field(
        default=1024, metadata={"help": "max chunk size"}
    )

    max_position_embeddings: Optional[int] = field(
        default=None, metadata={"help": "max position embeddings"}
    )

    access_token: Optional[str] = field(
        default=None, metadata={"help": "hf_access_token"}
    )

    lora_extra_params: Optional[str] = field(
        default=None, metadata={"help": "lora_extra_params"}
    )

    load_in_4_bit: Optional[bool] = field(
        default=False, metadata={"help": "lora_extra_params"}
    )

    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )

    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )

    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )

    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )

    