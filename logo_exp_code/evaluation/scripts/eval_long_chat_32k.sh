SAVE_PATH='./longbench/longchat-7b-v1.5-32k'
MODEL_NAME='llama-2'
MAX_LENGTH_SETTING='normal_setting'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python gen_longbench.py \
    --model_path='/vepfs/wcf/G/zecheng/hf_models/longchat-7b-v1.5-32k' \
    --model_max_length_setting=${MAX_LENGTH_SETTING} \
    --max_training_length=32768 \
    --save_path ${SAVE_PATH} \
    --model_name ${MODEL_NAME};

echo "generate finish ..., begin to evaluate ..."

python eval_longbench.py --pred_path=${SAVE_PATH}
