# DEEPSPEED_CONFIG="training/config/stage2.json"
# DDP_CONFIG="training/config/ddp.yaml"
ZERO1="/data/xys/szc/Retrieval_Head/training/config/zero1.json"
ZERO2="/data/xys/szc/Retrieval_Head/training/config/zero2.json"
ZERO3="/data/xys/szc/Retrieval_Head/training/config/zero3.json"
# 对应文件copy
FAST_ZERO3="/data/xys/szc/Retrieval_Head/training/config/zero3-fast.json"
DIR="/data/xys/szc/ckpt" #model存放地址


deepspeed --include localhost:3,4,5,6 training/train_simpo2_16k.py \
    --output_dir=${DIR}/simpo-llama2 \
    --max_steps=4000 \
    --logging_steps=1 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing=True \
    --learning_rate=1e-5 \
    --model_name_or_path="/data/xys/szc/models/llama2-7b-instruct" \
    --lr_scheduler_type="cosine" \
    --max_length=10000 \
    --max_position_embeddings=8192 \
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
    --dataset_path="/data/xys/szc/data/0730gen_data" \
    --full_finetune False \
    --remove_unused_columns False \
    --bf16 True \
    --rope_type "linear" \
    --rope_factor 8 \
    --beta 2.0 \
    --sft_weight 1 \
    --gamma_beta_ratio 0.25;
