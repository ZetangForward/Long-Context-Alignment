SAVE_PATH='./longbench/v3-aug'
MODEL_NAME='llama3'
MAX_LENGTH_SETTING='normal_setting'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python gen_longbench.py \
    --model_path='/vepfs/wcf/G/zecheng/ckpt/v3' \
    --peft_path='/vepfs/wcf/G/zecheng/ckpt/aug_sft_v3' \
    --max_position_embeddings=81920 \
    --model_max_length_setting=${MAX_LENGTH_SETTING} \
    --max_training_length=16384 \
    --save_path ${SAVE_PATH} \
    --model_name ${MODEL_NAME} \
    --use_logn;

echo "generate finish ..., begin to evaluate ..."

python eval_longbench.py --pred_path=${SAVE_PATH}