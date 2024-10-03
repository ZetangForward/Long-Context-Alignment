DEEPSPEED_CONFIG="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/stage3.yaml"
DDP_CONFIG="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/ddp.yaml"
ZERO2="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/stage2.json"
MEGATRON_CONFIG="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/megatron.yaml"

num_processes=1
echo "Number of processes to run: $num_processes"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --num_gpus 8 ./training/sft_train.py \
    --model_name_or_path="/vepfs/wcf/G/zecheng/hf_models/llama-2-7b-80k" \
    --peft_model_path="/vepfs/wcf/G/zecheng/ckpt/simpo_llama_10k_random_5/checkpoint-300/trainable" \
    --bf16=True \
    --output_dir="/vepfs/wcf/G/zecheng/ckpt/llama-2-80k-simpo-instruct-lora" \
    --dataset_path="/vepfs/wcf/G/zecheng/data/LongAlpaca-16k-length/LongAlpaca-16k-length.json" \
    --cache_dir="/vepfs/wcf/G/zecheng/cache" \
    --use_flash_attn True \
    --low_rank_training True \
    --num_train_epochs 1  \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 3 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 5e-5 \
    --weight_decay 0.0 \
    --model_max_length 16384 \
    --warmup_steps 20 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 1 \
    --max_steps 1000 \
    --deepspeed "/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/stage2.json" \
    --report_to "tensorboard" \
    --rope_type="linear" \
    --rope_theta=10000.0 \
    --remove_unused_columns=False \
    --factor=16;