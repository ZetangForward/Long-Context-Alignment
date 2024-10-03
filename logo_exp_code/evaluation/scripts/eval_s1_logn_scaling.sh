SAVE_PATH="./longbench"
CONTEXT_SETTING="normal_setting"
MODEL_NAME=simpo-8k-ds-${CONTEXT_SETTING}-logn_scaling
MODEL_PATH="/data/zecheng/ckpt/s1/merge_ckpt"
PEFT_PATH="/home/export/base/ycsc_lijt1/lijt1/online1/zecheng/ckpt/simpo-8k-ds/checkpoint-2000/context_scaling"

CUDA_VISIBLE_DEVICES=4 python gen_longbench.py \
    --model_path=${MODEL_PATH} \
    --max_position_embeddings=65536 \
    --rope_theta=200e6 \
    --model_max_length_setting=${CONTEXT_SETTING} \
    --save_path=${SAVE_PATH}/${MODEL_NAME} \
    --model_name=${MODEL_NAME};

echo "generate finish ..., begin to evaluate ..."

python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}
