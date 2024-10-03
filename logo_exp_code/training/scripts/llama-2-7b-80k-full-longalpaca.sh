DEEPSPEED_CONFIG="/data/zecheng/Retrieval_Head/training/config/stage2.json"
DEEPSPEED_CONFIG="/data/zecheng/Retrieval_Head/training/config/stage2.json"
ZERO2="/data/zecheng/Retrieval_Head/training/config/stage2.json"
ZERO3="/data/zecheng/Retrieval_Head/training/config/stage3.json"
DIR="/data/zecheng/ckpt"

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 --nnodes=1 ./training/sft_train.py \
        --model_name_or_path /vepfs/wcf/G/zecheng/hf_models/llama-2-7b-80k \
        --bf16 True \
        --output_dir "/data/zecheng/ckpt/s1/sft" \
        --model_max_length 65536 \
        --use_flash_attn True \
        --low_rank_training False \
        --gradient_checkpointing=True \
        --num_train_epochs 1  \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 200 \
        --save_total_limit 2 \
        --learning_rate 2e-5 \
        --warmup_steps 20 \
        --lr_scheduler_type "constant_with_warmup" \
        --logging_steps 1 \
        --max_steps 2000 \
        --rope_theta 200e6 \
        --save_only_model \
        --report_to "tensorboard";