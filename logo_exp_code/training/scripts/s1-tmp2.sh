DEEPSPEED_CONFIG="/data/zecheng/Retrieval_Head/training/config/stage2.json"
DEEPSPEED_CONFIG="/data/zecheng/Retrieval_Head/training/config/stage2.json"
ZERO2="/data/zecheng/Retrieval_Head/training/config/stage2.json"
ZERO3="/data/zecheng/Retrieval_Head/training/config/stage3.json"
DIR="/data/zecheng/ckpt"

deepspeed --num_gpus=8 training/train_simpo2.py \
    --output_dir=${DIR}/s1-tmp2-no-rope-type \
    --max_steps=4000 \
    --logging_steps=10 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing=True \
    --learning_rate=1e-4 \
    --model_name_or_path="/data/zecheng/hf_models/Meta-Llama-3-8B-Instruct" \
    --lr_scheduler_type="cosine" \
    --max_length=10000 \
    --max_position_embeddings=65536 \
    --max_prompt_length=9012 \
    --model_dtype="bfloat16" \
    --warmup_steps=50 \
    --save_steps=150 \
    --lora_r=32 \
    --lora_alpha=16 \
    --dataset_num_proc=16 \
    --save_total_limit=5 \
    --save_strategy="steps" \
    --evaluation_strategy="steps" \
    --eval_steps=150 \
    --load_best_model_at_end=True \
    --weight_decay=0.05 \
    --optim="paged_adamw_32bit" \
    --report_to="tensorboard" \
    --dataset_path="/data/zecheng/data/process_wiki_document/two_hop/hf_dataset_tmp2" \
    --full_finetune False \
    --save_only_model \
    --deepspeed=${ZERO3} \
    --save_safetensors=False \
    --rope_theta 200e6;
