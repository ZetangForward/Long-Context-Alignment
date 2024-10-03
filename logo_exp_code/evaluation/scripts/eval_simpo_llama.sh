SAVE_PATH="/vepfs/wcf/G/zecheng/Retrieval_Head/quick_eval/longbench"
MODEL_NAME="llama2-7b-random-pos-simpo-64k-2"

CUDA_VISIBLE_DEVICES=0,1 python gen_longbench.py \
    --model_path "/vepfs/wcf/G/zecheng/hf_models/Llama-2-7b-hf" \
    --peft_path "/vepfs/wcf/G/zecheng/ckpt/simpo_llama_10k_random_5/checkpoint-300/trainable" \
    --save_path ${SAVE_PATH} \
    --rope_theta=10000.0 \
    --rope_factor=16 --rope_type="linear" \
    --model_name ${MODEL_NAME};

echo "generate finish ..., begin to evaluate ..."

python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}