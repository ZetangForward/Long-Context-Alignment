DEEPSPEED_CONFIG="/data/zecheng/Retrieval_Head/training/config/stage2.json"
DEEPSPEED_CONFIG="/data/zecheng/Retrieval_Head/training/config/stage2.json"
DDP_CONFIG="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/ddp.yaml"
ZERO2="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/stage2.json"
DIR="/data/zecheng/ckpt"

CUDA_VISIBLE_DEVICES=0 python training/train_simpo2-qlora.py \
    --output_dir=${DIR}/s2 \
    --max_steps=4000 \
    --logging_steps=10 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing=False \
    --learning_rate=1e-4 \
    --model_name_or_path=/data/zecheng/hf_models/Meta-Llama-3-8B-Instruct \
    --lr_scheduler_type="cosine" \
    --max_length=10000 \
    --max_position_embeddings=65536 \
    --max_prompt_length=9012 \
    --bits 4 \
    --warmup_steps=100 \
    --save_steps=50 \
    --lora_r=32 \
    --lora_alpha=16 \
    --dataset_num_proc=16 \
    --save_total_limit=5 \
    --save_strategy="steps" \
    --evaluation_strategy="steps" \
    --eval_steps=50 \
    --load_best_model_at_end True \
    --weight_decay=0.05 \
    --optim="paged_adamw_32bit" \
    --report_to="tensorboard" \
    --dataset_path="/data/zecheng/data/process_wiki_document/two_hop/hf_dataset_tmp2" \
    --save_only_model \
    --rope_theta 200e6;
    
