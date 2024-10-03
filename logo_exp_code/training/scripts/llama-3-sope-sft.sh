DEEPSPEED_CONFIG="training/config/stage2.json"
DEEPSPEED_CONFIG="training/config/stage2.json"
ZERO2="training/config/stage2.json"
ZERO3="training/config/stage3.json"
ZERO3_FAST="training/config/zero3-fast.json"
DIR="/vepfs/wcf/G/zecheng/ckpt"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_LAUNCH_BLOCKING=1 deepspeed --num_gpus=8 ./training/train_sope-language-modeling.py  \
        --output_dir=${DIR}/s1-sope \
        --max_steps=10000 \
        --num_train_epochs=3.0 \
        --logging_steps=10 \
        --per_device_train_batch_size=3 \
        --per_device_eval_batch_size=3 \
        --gradient_accumulation_steps=4 \
        --gradient_checkpointing=True \
        --learning_rate=1e-4 \
        --model_name_or_path="/data/zecheng/hf_models/Meta-Llama-3-8B-Instruct" \
        --lr_scheduler_type="cosine" \
        --max_length=10000 \
        --bf16 True \
        --max_position_embeddings=65536 \
        --model_dtype="bfloat16" \
        --warmup_steps=50 \
        --save_steps=200 \
        --lora_r=32 \
        --lora_alpha=16 \
        --dataset_num_proc=32 \
        --save_total_limit=5 \
        --save_strategy="steps" \
        --evaluation_strategy="no" \
        --weight_decay=0.05 \
        --optim="paged_adamw_32bit" \
        --report_to="tensorboard" \
        --dataset_path="/data/zecheng/data/processed_project/step4_jsonl_data_less" \
        --deepspeed=${ZERO2} \
        --save_only_model \
        --load_in_4_bit=False \
        --save_safetensors=False \
        --rope_theta 200e6;