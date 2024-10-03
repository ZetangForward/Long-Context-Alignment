DEEPSPEED_CONFIG="training/config/stage2.json"
DDP_CONFIG="training/config/ddp.yaml"
ZERO2="training/config/stage2.json"
ZERO3="./training/config/stage3.json"
FAST_ZERO3="./training/config/zero3-fast.json"
DIR="/data/zecheng/ckpt"

deepspeed --num_gpus=8 training/train_simpo2_16k.py \
    --output_dir=${DIR}/simpo-16k \
    --max_steps=6000 \
    --logging_steps=1 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing=True \
    --learning_rate=1e-4 \
    --model_name_or_path="/data/zecheng/hf_models/Meta-Llama-3-8B-Instruct" \
    --lr_scheduler_type="cosine" \
    --max_length=10000 \
    --max_position_embeddings=65536 \
    --max_prompt_length=9012 \
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
    --optim="paged_adamw_32bit" \
    --report_to="tensorboard" \
    --dataset_path="/data/zecheng/data/processed_project/mix_chunks_v3/combine" \
    --deepspeed=${ZERO2} \
    --full_finetune True \
    --remove_unused_columns False \
    --rope_theta 200e6 \
    --bf16 True \
    --beta 2.5 \
    --sft_weight 0.0 \
    --gamma_beta_ratio 0.55;



# 
