SAVE_PATH="/vepfs/wcf/G/zecheng/Retrieval_Head/quick_eval/longbench"
SAVE_PATH="/vepfs/wcf/G/zecheng/Retrieval_Head/quick_eval/longbench"
MODEL_NAME="Llama3-8b-80k-sft-alpaca"

CUDA_VISIBLE_DEVICES=0,1 python gen_longbench.py \
    --model_path "/data/zecheng/ckpt/Llama-3-8B-Instruct-80K-QLoRA-Merged-sft/checkpoint-600" \
    --save_path ${SAVE_PATH} \
    --model_name ${MODEL_NAME};

echo "generate finish ..., begin to evaluate ..."

python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}
