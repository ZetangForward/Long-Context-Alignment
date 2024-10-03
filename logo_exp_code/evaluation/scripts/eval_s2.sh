SAVE_PATH="./longbench"
MODEL_NAME="s2/s2"
CUDA_VISIBLE_DEVICES=7 python gen_longbench.py \
    --model_path="/data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged" \
    --peft_path="/data/zecheng/ckpt/s2/checkpoint-1900/context_scaling" \
    --rope_theta 200e6 \
    --save_path=${SAVE_PATH}/${MODEL_NAME} \
    --model_name=${MODEL_NAME};

echo "generate finish ..., begin to evaluate ..."

# python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}
