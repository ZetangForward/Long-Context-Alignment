SAVE_PATH="/data/zecheng/Retrieval_Head/quick_eval/fanoutqa"
MODEL_NAME="Llama3-8b-80k-Instruct"

CUDA_VISIBLE_DEVICES=1 python gen_fanout_qa.py \
    --model_path "/data/zecheng/ckpt/Llama-3-8B-Instruct-80K-QLoRA-Merged-sft/checkpoint-600" \
    --save_path ${SAVE_PATH} \
    --model_name ${MODEL_NAME};
