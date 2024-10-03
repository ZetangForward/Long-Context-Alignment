DEEPSPEED_CONFIG="training/config/stage2.json"
DEEPSPEED_CONFIG="training/config/stage2.json"
ZERO2="training/config/stage2.json"
ZERO3="training/config/stage3.json"
ZERO3_FAST="training/config/zero3-fast.json"
DIR="/data/zecheng/ckpt"

deepspeed --num_gpus=8 training/train_sope-language-modeling.py \
    --output_dir=${DIR}/v3_aug \
    --max_steps=1000 \
    --logging_steps=10 \
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
    --save_only_model \
    --lora_extra_params="embed_tokens" \
    --load_in_4_bit=False \
    --deepspeed=${ZERO3_FAST} \
    --save_safetensors=False \
    --rope_theta 200e6;


