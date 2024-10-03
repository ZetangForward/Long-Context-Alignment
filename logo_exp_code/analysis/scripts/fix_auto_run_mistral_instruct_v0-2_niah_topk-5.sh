#!/bin/bash

# 函数：检查GPU是否空闲并运行任务
run_if_gpu_free() {
    local device=$1
    local needle_id=$2
    
    # 检查GPU是否空闲
    if nvidia-smi -i $device | grep -q "No running processes found"; then
        echo "GPU $device is free. Running task for needle_id $needle_id."
        
        # 运行Python脚本
        CUDA_VISIBLE_DEVICES=$device python retrieval_head_detection_NIAH.py \
            --model_path /data/zecheng/hf_models/Mistral-7B-Instruct-v0.2 \
            --s 0 --e 50000 --needle_id $needle_id --model_name_suffix "NIAH_fix_topk-5" --topk 5 \
            --needle_file "needles.jsonl" > ./logs/mistral_niah-$needle_id.log 2>&1 &
        
        echo "Task started on GPU $device for needle_id $needle_id"
        return 0  # 任务成功启动
    else
        echo "GPU $device is busy."
        return 1  # GPU忙碌，未能启动任务
    fi
}

# 函数：检查特定任务是否正在运行
is_task_running() {
    local needle_id=$1
    if ps aux | grep -v grep | grep -q "python.*needle_id $needle_id"; then
        return 0  # 任务正在运行
    else
        return 1  # 任务没有运行
    fi
}

# 主循环
first_group_started=false
second_group_started=false

while true; do
    if ! $first_group_started; then
        # 尝试启动第一组任务
        run_if_gpu_free 0 7 && run_if_gpu_free 1 8 && run_if_gpu_free 2 9 && run_if_gpu_free 3 10
        
        # 检查是否所有第一组任务都已启动
        if is_task_running 7 && is_task_running 8 && is_task_running 9 && is_task_running 10; then
            first_group_started=true
            echo "All tasks in the first group have been started."
        fi
    elif $first_group_started && ! $second_group_started; then
        # 检查第一组任务是否都已完成
        if ! (is_task_running 7 || is_task_running 8 || is_task_running 9 || is_task_running 10); then
            echo "All tasks in the first group are completed. Starting second group."
            
            # 尝试启动第二组任务
            run_if_gpu_free 0 3 && run_if_gpu_free 1 4 && run_if_gpu_free 2 5 && run_if_gpu_free 3 6
            
            # 检查是否所有第二组任务都已启动
            if is_task_running 3 && is_task_running 4 && is_task_running 5 && is_task_running 6; then
                second_group_started=true
                echo "All tasks in the second group have been started."
                break  # 退出主循环
            fi
        fi
    fi

    # 等待一段时间再次检查
    sleep 300  # 每5分钟检查一次
done

echo "All tasks have been initiated. Script is exiting."
