DEEPSPEED_CONFIG="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/stage2.json"
DEEPSPEED_CONFIG="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/stage2.json"
ZERO2="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/stage2.json"
ZERO3="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/stage3.json"
DIR="/vepfs/wcf/G/zecheng/ckpt"

deepspeed --num_gpus=8 training/train_simpo2.py \
    --output_dir=${DIR}/s1 \
    --max_steps=1000 \
    --logging_steps=10 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing=True \
    --learning_rate=1e-4 \
    --peft_model_path="/vepfs/wcf/G/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA" \
    --model_name_or_path=/vepfs/wcf/G/zecheng/hf_models/Meta-Llama-3-8B-Instruct \
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
    --dataset_path="/vepfs/wcf/G/zecheng/data/hf_dataset_tmp" \
    --save_only_model \
    --deepspeed=${ZERO3} \
    --save_only_model \
    --save_safetensors=False \
    --rope_theta 200e6;
