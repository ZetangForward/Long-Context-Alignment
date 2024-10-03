#!/bin/bash

# Define the base model path and other constants
BASE_MODEL_PATH="/vepfs/wcf/G/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged"
PEFT_BASE_PATH='/vepfs/wcf/G/zecheng/ckpt/llama3_instruct_tuning'
LOG_DIR="./logs"

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

# Function to check GPU availability specifically for retrieval_head_peft_NIAH.py script
check_gpu_availability() {
    while true; do
        # Check if the specific python script is running
        if ! pgrep -f "retrieval_head_peft_NIAH.py" > /dev/null; then
            echo "GPUs are free now."
            break
        else
            echo "GPUs are busy, waiting..."
            sleep 60
        fi
    done
}


# Function to run experiments on all 8 GPUs
run_experiments() {
    local step=$1
    local model_name_suffix="instruct-training-step-$step"
    local peft_model_path="$PEFT_BASE_PATH/checkpoint-$step"

    for gpu in {0..7}; do
        local needle_id=$((gpu + 3))
        local cmd="CUDA_VISIBLE_DEVICES=$gpu python retrieval_head_peft_NIAH.py \
            --model_path $BASE_MODEL_PATH \
            -di 10 -ci 20 -s 0 -e 50000 --needle_id $needle_id --model_name_suffix \"$model_name_suffix\" \
            --needle_file \"needles.jsonl\" --peft_model_path '$peft_model_path' --topk 1 > $LOG_DIR/llama3_niah-$model_name_suffix-id-$needle_id.log 2>&1 &"
        
        echo "Running on GPU $gpu: $cmd"
        eval $cmd
    done
    # Wait for all background jobs to complete
    wait
}

# Main loop
for step in {100..1000..200}; do
    echo "Starting experiments for step $step"
    run_experiments $step
    check_gpu_availability
done

echo "All tasks completed."
