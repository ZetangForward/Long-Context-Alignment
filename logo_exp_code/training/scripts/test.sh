DEEPSPEED_CONFIG="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/stage3.yaml"
DDP_CONFIG="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/ddp.yaml"
ZERO2="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/stage2.json"
MEGATRON_CONFIG="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/megatron.yaml"

num_processes=1
echo "Number of processes to run: $num_processes"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --num_gpus 8 training/train_simpo.py \
    --output_dir="/vepfs/wcf/G/zecheng/ckpt/simpo_llama_10k_random_5" \
    --max_steps=3000 \
    --logging_steps=1 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing=True \
    --learning_rate=1e-4 \
    --model_name_or_path "/vepfs/wcf/G/zecheng/hf_models/Llama-2-7b-hf" \
    --lr_scheduler_type="cosine" \
    --max_length=10000 \
    --eval_strategy="steps" \
    --max_prompt_length=9012 \
    --max_chunk_size=256 \
    --max_qa_size=1024 \
    --model_dtype="bfloat16" \
    --warmup_steps=100 \
    --save_steps=100 \
    --eval_steps=100 \
    --dataset_num_proc=1 \
    --save_total_limit=5 \
    --save_strategy="steps" \
    --weight_decay=0.05 \
    --lora_r=32 \
    --lora_alpha=16 \
    --optim="paged_adamw_32bit" \
    --report_to="tensorboard" \
    --dataset_path="/vepfs/wcf/G/zecheng/data/SlimPajama-6B/dpo_data/hf_data_64" \
    --gamma=0.5 \
    --load_best_model_at_end=True \
    --rope_type="linear" \
    --rope_theta=10000.0 \
    --factor=16;