ZERO1="./training/config/zero1.json"
ZERO2="./training/config/zero2.json"
ZERO3="./training/config/zero3.json"
FAST_ZERO3="./training/config/zero3-fast.json"
SAVE_DIR="/vepfs/wcf/G/zecheng/ckpt" # Attention!!!! 这里是model存放地址，需要及时修改
DATA_DIR='/vepfs/wcf/G/zecheng/data'
MODEL_DIR='/vepfs/wcf/G/zecheng/hf_models'

deepspeed --include localhost:0,1,2,3,4,5,6,7 training/language_modeling.py \
    --output_dir=${SAVE_DIR}/llama3_post_train_langauge_modeling \
    --max_steps=2000 \
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
    --weight_decay=0.05 \
    --deepspeed=${ZERO2} \
    --optim="paged_adamw_32bit" \
    --report_to="tensorboard" \
    --dataset_path=${DATA_DIR}/'SlimPajama-6B-tokenized-llama3-32768' \
    --remove_unused_columns False \
    --bf16 True \
    --low_rank_training True