DEEPSPEED_CONFIG="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/stage3.yaml"
DDP_CONFIG="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/ddp.yaml"
ZERO2="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/stage2.json"
ZERO3="/vepfs/wcf/G/zecheng/Retrieval_Head/training/config/stage3.json"
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
num_processes=8
echo "Number of processes to run: $num_processes"

accelerate launch --mixed_precision bf16 --num_processes ${num_processes} \
    --use_deepspeed --deepspeed_config_file ${ZERO3} \
    training/train.py \
    --output_dir="/vepfs/wcf/G/zecheng/ckpt/dpo-llama" \
    --max_steps=4000 \
    --logging_steps=10 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing=False \
    --learning_rate=1e-4 \
    --model_name_or_path "/vepfs/wcf/G/zecheng/hf_models/Llama-2-7b-hf" \
    --lr_scheduler_type="cosine" \
    --max_length=9000 \
    --eval_strategy="steps" \
    --max_prompt_length=8192 \
    --model_dtype "bfloat16" \
    --warmup_steps=100 \
    --save_steps=200 \
    --dataset_num_proc 16 \
    --save_total_limit=5 \
    --save_strategy="steps" \
    --weight_decay=0.05 \
    --optim="paged_adamw_32bit" \
    --report_to="tensorboard" \
    --dataset_path="/vepfs/wcf/G/zecheng/data/SlimPajama-6B/dpo_data/hf_data_v2" \
    --run_name "dpo_llama_2" \
    --load_best_model_at_end True;