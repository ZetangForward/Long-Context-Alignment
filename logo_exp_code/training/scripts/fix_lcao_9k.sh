DEEPSPEED_CONFIG="training/config/stage2.json"
DDP_CONFIG="training/config/ddp.yaml"
ZERO2="training/config/stage2.json"
ZERO3="./training/config/stage3.json"
FAST_ZERO3="./training/config/zero3-fast.json"
DIR="/data/zecheng/ckpt"

CUDA_VISIBLE_DEVICES=0 python training/train_simpo2_16k.py \
    --output_dir=${DIR}/simpo-16-mix-v3-lora \
    --max_steps=10000 \
    --logging_steps=1 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing=True \
    --learning_rate=1e-4 \
    --model_name_or_path="/data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged" \
    --lr_scheduler_type="cosine" \
    --max_position_embeddings=65536 \
    --model_dtype="bfloat16" \
    --warmup_steps=100 \
    --save_steps=200 \
    --lora_r=32 \
    --lora_alpha=16 \
    --dataset_num_proc=16 \
    --save_total_limit=5 \
    --save_strategy="steps" \
    --evaluation_strategy="steps" \
    --eval_steps=200 \
    --load_best_model_at_end True \
    --weight_decay=0.05 \
    --optim="paged_adamw_32bit" \
    --report_to="tensorboard" \
    --deepspeed=${FAST_ZERO3} \
    --dataset_path="/data/zecheng/data/processed_project/mix_chunks_v3/combine" \
    --save_only_model \
    --full_finetune False \
    --remove_unused_columns False \
    --rope_theta 200e6 \
    --beta 2.5 \
    --sft_weight 0.1 \
    --gamma_beta_ratio 0.55;
    

# deepspeed --num_gpus=8
