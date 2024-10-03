ZERO1="./config/zero1.json"
ZERO2="./config/zero2.json"
ZERO3="./config/zero3.json"
FAST_ZERO3="./config/zero3-fast.json"
SAVE_DIR="/public/home/zecheng/workspace/zecheng" # Attention!!!! 这里是model存放地址，需要及时修改
DATA_DIR='/public/home/zecheng/workspace/zecheng/data'
MODEL_DIR='/public/home/zecheng/workspace/hf_models'

deepspeed --include localhost:0,1,2,3,4,5,6,7 ./Instruct_tuning.py \
    --output_dir=${SAVE_DIR}/llama3_instruct_tuning \
    --max_steps=1000 \
    --logging_steps=1 \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing=True \
    --learning_rate=1e-5 \
    --model_name_or_path=${MODEL_DIR}/'Llama-3-8B-Instruct-80K-QLoRA-Merged' \
    --lr_scheduler_type="cosine" \
    --max_position_embeddings=81920 \
    --warmup_steps=100 \
    --save_steps=50 \
    --lora_r=32 \
    --lora_alpha=16 \
    --save_total_limit=20 \
    --save_strategy='steps' \
    --eval_strategy='no' \
    --weight_decay=0.05 \
    --deepspeed=${ZERO2} \
    --optim="paged_adamw_32bit" \
    --report_to="tensorboard" \
    --dataset_path=${DATA_DIR}/'LongAlpaca-12k' \
    --remove_unused_columns False \
    --bf16 True \
    --low_rank_training True