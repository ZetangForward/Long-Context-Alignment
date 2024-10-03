DEEPSPEED_CONFIG="training/config/stage2.json"
DDP_CONFIG="training/config/ddp.yaml"
ZERO1="training/config/zero1.json"
ZERO2="training/config/stage2.json"
ZERO3="./training/config/stage3.json"
FAST_ZERO3="./training/config/zero3-fast.json"
DIR="/nvme/zecheng/ckpt"

deepspeed --num_gpus=8 training/train_simpo2_llama_16k.py \
    --output_dir=${DIR}/simpo-llama2_fix \
    --max_steps=6000 \
    --logging_steps=1 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing=True \
    --learning_rate=1e-5 \
    --model_name_or_path="/data/zecheng/hf_models/longchat-7b-v1.5-32k" \
    --lr_scheduler_type="cosine" \
    --max_length=10000 \
    --max_position_embeddings=4096 \
    --model_dtype "bfloat16" \
    --warmup_steps=100 \
    --save_steps=100 \
    --lora_r=32 \
    --lora_alpha=16 \
    --dataset_num_proc=16 \
    --save_total_limit=5 \
    --save_strategy="steps" \
    --evaluation_strategy="steps" \
    --eval_steps=100 \
    --load_best_model_at_end True \
    --weight_decay=0.05 \
    --deepspeed=${ZERO1} \
    --optim="paged_adamw_32bit" \
    --report_to="tensorboard" \
    --dataset_path="/nvme/zecheng/data/train_data/v1-32k" \
    --full_finetune False \
    --remove_unused_columns False \
    --bf16 True \
    --rope_type "linear" \
    --rope_factor 8 \
    --beta 2.0 \
    --sft_weight 1 \
    --gamma_beta_ratio 0.25;
