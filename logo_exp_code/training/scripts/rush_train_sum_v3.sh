# DEEPSPEED_CONFIG="training/config/stage2.json"
# DDP_CONFIG="training/config/ddp.yaml"
ZERO1="training/config/zero1.json"
ZERO2="./training/config/config/zero2.json"
ZERO3="./training/config/config/zero3.json"
# 对应文件copy
FAST_ZERO3="./training/config/config/zero3-fast.json"
DIR="/vepfs/wcf/G/zecheng/ckpt" #model存放地址

deepspeed --include localhost:0,1,2,3,4,5,6,7 ./training/sft_rush_train.py \
    --output_dir=${DIR}/aug_sft_v3 \
    --max_steps=1000 \
    --logging_steps=1 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing=True \
    --learning_rate=1e-5 \
    --model_name_or_path="/vepfs/wcf/G/zecheng/ckpt/v3" \
    --lr_scheduler_type="cosine" \
    --max_position_embeddings=81920 \
    --warmup_steps=100 \
    --save_steps=200 \
    --lora_r=32 \
    --lora_alpha=16 \
    --save_total_limit=5 \
    --save_strategy="no" \
    --weight_decay=0.05 \
    --deepspeed=${ZERO1} \
    --optim="paged_adamw_32bit" \
    --report_to="tensorboard" \
    --dataset_path="/vepfs/wcf/G/zecheng/data/v3-aug" \
    --remove_unused_columns False \
    --bf16 True \
    --rope_theta 200e6 \
    --low_rank_training True

