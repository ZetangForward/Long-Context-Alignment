SAVE_PATH="./longbench"
MODEL_NAME="LongAlpaca-7B"

CUDA_VISIBLE_DEVICES=6 python gen_longbench.py \
    --model_path "/data/zecheng/hf_models/LongAlpaca-7B" \
    --save_path ${SAVE_PATH}/${MODEL_NAME} \
    --model_max_length_setting "normal_setting" \
    --model_name ${MODEL_NAME};

echo "generate finish ..., begin to evaluate ..."

python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}
