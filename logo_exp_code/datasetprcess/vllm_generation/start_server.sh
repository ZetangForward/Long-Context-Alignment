#!/bin/bash

# 检查是否提供了参数（模型路径）
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi

# 将第一个命令行参数赋值给MODEL_PATH变量
MODEL_PATH=$1

# 使用for循环启动8个服务实例，每个实例使用不同的GPU和端口
for i in {0..7}
do
    sleep 5
    CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.api_server \
        --model $MODEL_PATH \
        --gpu-memory-utilization=0.9 \
        --max-num-seqs=200 \
        --host 127.0.0.1 --tensor-parallel-size 1 \
        --port $((4100+i)) \
        --swap-space 0 \
    &
done

# bash start_server.sh /data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged
# bash start_server.sh /data/zecheng/hf_models/llama-2-7b-80k
# bash start_server.sh /data/zecheng/hf_models/longchat-7b-v1.5-32k
# bash start_server.sh /public/home/ljt/wpz/RLHF/rlhflow/AgentPlay/Ablations-llama2/EI_2play/models/EI_from_llama3_8b/checkpoint-174
# pkill -f vllm.entrypoints.api_server