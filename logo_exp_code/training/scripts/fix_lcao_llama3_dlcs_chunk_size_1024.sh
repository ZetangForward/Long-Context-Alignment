ZERO1="./config/zero1.json"
ZERO2="./config/zero2.json"
ZERO3="./config/zero3.json"
FAST_ZERO3="./config/zero3-fast.json"
SAVE_DIR="/public/home/zecheng/workspace/zecheng/ckpt" # Attention!!!! 这里是model存放地址，需要及时修改
DATA_DIR='/public/home/zecheng/workspace/zecheng/data/iclr2025'
MODEL_DIR='/public/home/zecheng/workspace/hf_models'

deepspeed --include localhost:0,1,2,3,4,5,6,7 ./lcao_train.py \
    --output_dir=${SAVE_DIR}/lcao_chunk_16_size_1024_change_lr \
    --max_steps=1200 \
    --logging_steps=1 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing=True \
    --learning_rate=1e-6 \
    --model_name_or_path=${MODEL_DIR}/'meta-llama-3-8b-instruct' \
    --lr_scheduler_type="cosine" \
    --max_position_embeddings=81920 \
    --warmup_steps=100 \
    --save_steps=50 \
    --lora_r=32 \
    --max_target_length=2000 \
    --lora_alpha=16 \
    --max_seq_length=13000 \
    --dataloader_num_workers=0 \
    --save_total_limit=10 \
    --save_strategy="steps" \
    --eval_strategy="steps" \
    --eval_steps=50 \
    --load_best_model_at_end False \
    --weight_decay=0.05 \
    --rope_theta=200000000.0 \
    --optim="paged_adamw_32bit" \
    --report_to="tensorboard" \
    --dataset_path=${DATA_DIR}/'chunk_16_size_1024' \
    --low_rank_training True \
    --remove_unused_columns False \
    --deepspeed=${ZERO3} \
    --bf16 True \
    --beta 2.5 \
    --sft_weight 0.5 \
    --gamma_beta_ratio 0.55;