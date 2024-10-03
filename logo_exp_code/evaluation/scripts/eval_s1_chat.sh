SAVE_PATH="./longbench"
CONTEXT_SETTING="long_setting"
MODEL_NAME=s1-${CONTEXT_SETTING}-chat

CUDA_VISIBLE_DEVICES=6 python gen_longbench.py \
    --model_path="/data/zecheng/ckpt/s1/merge_ckpt" \
    --max_position_embeddings=65536 \
    --rope_theta 200e6 \
    --model_max_length_setting="long_setting" \
    --save_path=${SAVE_PATH}/${MODEL_NAME} \
    --model_name=${MODEL_NAME};

echo "generate finish ..., begin to evaluate ..."

python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}
