SAVE_PATH='./longbench/llama-2-simpo'
MODEL_NAME='llama-2'
MAX_LENGTH_SETTING='normal_setting'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python gen_longbench.py \
    --model_path='/data/zecheng/hf_models/longchat-7b-v1.5-32k' \
    --peft_path='/nvme/zecheng/ckpt/simpo-llama2_fix/checkpoint-700/context_scaling' \
    --model_max_length_setting=${MAX_LENGTH_SETTING} \
    --max_training_length=16384 \
    --save_path=${SAVE_PATH} \
    --model_name=${MODEL_NAME} \
    --use_logn;

echo "generate finish ..., begin to evaluate ..."

python eval_longbench.py --pred_path=${SAVE_PATH}
