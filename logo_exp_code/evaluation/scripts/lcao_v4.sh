SAVE_PATH='./longbench/lcao_v4'
MODEL_NAME='llama3'
MAX_LENGTH_SETTING='normal_setting'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python gen_longbench.py \
    --model_path='/vepfs/wcf/G/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged' \
    --peft_path='/vepfs/wcf/G/zecheng/ckpt/simpo_v4/checkpoint-300' \
    --max_position_embeddings=81920 \
    --model_max_length_setting=${MAX_LENGTH_SETTING} \
    --save_path=${SAVE_PATH} \
    --model_type=${MODEL_NAME};

echo "generate finish ..., begin to evaluate ..."

python eval_longbench.py --pred_path=${SAVE_PATH}