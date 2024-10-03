DEEPSPEED_CONFIG="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/stage3.yaml"
DDP_CONFIG="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/ddp.yaml"
ZERO2="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/stage2.json"
MEGATRON_CONFIG="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/megatron.yaml"
# export NCCL_IB_DISABLE=1
# export NCCL_NET_GDR_LEVEL=0
# export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=0

# 获取 CUDA_VISIBLE_DEVICES 环境变量的值
# cuda_devices=$(echo $CUDA_VISIBLE_DEVICES)

# 将逗号分隔的值转换为数组
# IFS=',' read -r -a device_array <<< "$cuda_devices"

# 计算进程数
# num_processes=${#device_array[@]}

# accelerate launch --mixed_precision bf16 --num_processes=1 \
# CUDA_VISIBLE_DEVICES=0,1 python training/train.py \
# CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --mixed_precision bf16 --num_processes=4 \

CUDA_VISIBLE_DEVICES=0,1 python training/train_simpo.py \
    --output_dir="/vepfs/wcf/G/zecheng/ckpt/simpo-llama-2-7b-80k" \
    --max_steps=4000 \
    --logging_steps=10 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing=False \
    --learning_rate=1e-4 \
    --model_name_or_path="/vepfs/wcf/G/zecheng/hf_models/llama-2-7b-80k" \
    --lr_scheduler_type="cosine" \
    --max_length=10000 \
    --eval_strategy="steps" \
    --max_prompt_length=9012 \
    --model_dtype="bfloat16" \
    --warmup_steps=100 \
    --save_steps=200 \
    --eval_steps=200 \
    --dataset_num_proc=16 \
    --save_total_limit=5 \
    --save_strategy="steps" \
    --weight_decay=0.05 \
    --optim="paged_adamw_32bit" \
    --report_to="tensorboard" \
    --dataset_path="/vepfs/wcf/G/zecheng/data/SlimPajama-6B/dpo_data/hf_data_v2" \
    --run_name="simpo-llama2-80k" \
    --load_best_model_at_end True \
    --gamma=0.5 \
    --factor=10.0 \
    --dynamic="dynamic" \
    --rope_theta=10000.0 \

