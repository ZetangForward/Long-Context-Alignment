# export CUDA_VISIBLE_DEVICES=0

deepspeed --num_gpus 4 ./training/raw_sft_train.py  \
        --model_name_or_path "/data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged" \
        --bf16 True \
        --output_dir "/data/zecheng/ckpt/Llama-3-8B-Instruct-80K-QLoRA-Merged-sft"       \
        --dataset_path "/data/zecheng/data/alpaca-gpt4" \
        --cache_dir "/vepfs/wcf/G/zecheng/cache" \
        --model_max_length 81920 \
        --tokenizer_max_length 4096 \
        --use_flash_attn True \
        --low_rank_training False \
        --num_train_epochs 1  \
        --per_device_train_batch_size 1     \
        --per_device_eval_batch_size 2     \
        --gradient_accumulation_steps 8     \
        --evaluation_strategy "no"     \
        --save_strategy "steps"     \
        --save_steps 200    \
        --save_total_limit 2     \
        --remove_unused_columns False \
        --learning_rate 2e-5     \
        --weight_decay 0.0     \
        --warmup_steps 20     \
        --lr_scheduler_type "constant_with_warmup"     \
        --logging_steps 1     \
        --deepspeed "/data/zecheng/Retrieval_Head/training/config/stage2.json" \
        --max_steps 1000 \
        --report_to "tensorboard";