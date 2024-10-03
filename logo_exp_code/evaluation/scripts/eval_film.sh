SAVE_PATH='./longbench/film-7b'
MODEL_NAME='mistral'
MAX_LENGTH_SETTING='normal_setting'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python gen_longbench.py \
    --model_path='/data/zecheng/hf_models/FILM-7B' \
    --peft_path='/nvme/zecheng/ckpt/simpo-FILM/checkpoint-200/context_scaling' \
    --max_position_embeddings=32768 \
    --model_max_length_setting=${MAX_LENGTH_SETTING} \
    --max_training_length=32768 \
    --save_path=${SAVE_PATH} \
    --model_name=${MODEL_NAME} \
    --use_logn;

echo "generate finish ..., begin to evaluate ..."

python eval_longbench.py --pred_path=${SAVE_PATH}
