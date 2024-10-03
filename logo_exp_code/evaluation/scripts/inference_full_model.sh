SAVE_PATH=$1
MODEL_PATH=$2
MODEL_TYPE=$3
MAX_LENGTH_SETTING=$4
MAX_LENGTH=$5

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python gen_longbench.py \
    --model_path=${MODEL_PATH} \
    --max_position_embeddings=${MAX_LENGTH} \
    --model_max_length_setting=${MAX_LENGTH_SETTING} \
    --save_path=${SAVE_PATH} \
    --model_type=${MODEL_TYPE};

echo "generate finish ..., begin to evaluate ..."

python eval_longbench.py --pred_path=${SAVE_PATH}


# ======== Example Usage ========

# bash scripts/inference_full_model.sh ./longbench/llama2-80k /vepfs/wcf/G/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged llama-2 normal_setting 81920
# bash scripts/inference_full_model.sh ./longbench/llama3-1-8b-instruct /data/zecheng/hf_models/llama3.1-8b-instruct llama-3 normal_setting -1
# bash scripts/inference_full_model.sh ./longbench/llama3-8B-Instruct-YaRN /data/zecheng/hf_models/Meta-Llama-3-8B-Instruct llama-3 normal_setting 81920 
# bash scripts/inference_full_model.sh ./longbench/llama3-8B-Instruct-Original /vepfs/wcf/G/zecheng/hf_models/Meta-Llama-3-8B-Instruct llama-3 8k_setting -1