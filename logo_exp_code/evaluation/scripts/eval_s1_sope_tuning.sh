SAVE_PATH="./longbench"
CONTEXT_SETTING="normal_setting"
MODEL_NAME=simpo-8k-${CONTEXT_SETTING}-sope-sft
MODEL_PATH="/data/zecheng/hf_models/Meta-Llama-3-8B-Instruct"
PEFT_PATH="/data/zecheng/ckpt/s1-sope-sft/checkpoint-1600/context_scaling"

CUDA_VISIBLE_DEVICES=6 python gen_longbench.py \
    --model_path=${MODEL_PATH} \
    --max_position_embeddings=65536 \
    --max_training_length 8192 \
    --rope_theta=200e6 \
    --model_max_length_setting=${CONTEXT_SETTING} \
    --save_path=${SAVE_PATH}/${MODEL_NAME} \
    --model_name=${MODEL_NAME};

echo "generate finish ..., begin to evaluate ..."

python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}
