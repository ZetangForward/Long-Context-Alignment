SAVE_PATH="/vepfs/wcf/G/zecheng/Retrieval_Head/quick_eval/longbench"
MODEL_NAME="longchat-7b-v1.5-32k"

CUDA_VISIBLE_DEVICES=7 python gen_longbench.py \
    --model_path "/data/zecheng/hf_models/longchat-7b-v1.5-32k" \
    --save_path ${SAVE_PATH}/${MODEL_NAME} \
    --model_max_length_setting "normal_setting" \
    --model_name ${MODEL_NAME};

echo "generate finish ..., begin to evaluate ..."

python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}
