SAVE_PATH=$1
MODEL_PATH=$2
MODEL_TYPE=$3
MAX_LENGTH_SETTING=$4
MAX_LENGTH=$5
PEFT_PATH=$6
ROPE_THETA=$7

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python gen_longbench.py \
    --model_path=${MODEL_PATH} \
    --max_position_embeddings=${MAX_LENGTH} \
    --peft_path=${PEFT_PATH} \
    --model_max_length_setting=${MAX_LENGTH_SETTING} \
    --save_path=${SAVE_PATH} \
    --rope_theta=${ROPE_THETA} \
    --model_type=${MODEL_TYPE};

echo "generate finish ..., begin to evaluate ..."

python eval_longbench.py --pred_path=${SAVE_PATH}


# ======== Example Usage ========
# bash scripts/inference_peft_model.sh ./longbench/lcao_chunk_16_size_1024 /vepfs/wcf/G/zecheng/hf_models/Meta-Llama-3-8B-Instruct llama-3 normal_setting 81920 /vepfs/wcf/G/zecheng/ckpt/lcao_chunk_16_size_1024/checkpoint-300 2e8
# bash scripts/inference_peft_model.sh ./longbench/fix_lcao_chunk_16_size_1024 /vepfs/wcf/G/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged llama-3 normal_setting -1 /vepfs/wcf/G/zecheng/ckpt/fix_lcao_chunk_16_size_1024/checkpoint-600/context_scaling -1
# bash scripts/inference_peft_model.sh ./longbench/pose_chunk_size_1024 /vepfs/wcf/G/zecheng/hf_models/Meta-Llama-3-8B-Instruct llama-3 normal_setting 81920 /vepfs/wcf/G/zecheng/ckpt/pose_chunk_size_1024/checkpoint-1000/context_scaling 200e6
# bash scripts/inference_peft_model.sh ./longbench/lcao_v1-5 /vepfs/wcf/G/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged llama-3 normal_setting -1 /vepfs/wcf/G/zecheng/ckpt/lcao_v1-5/checkpoint-500
# bash scripts/inference_peft_model.sh ./longbench/lcao_v6_ckpt_500 /vepfs/wcf/G/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged llama-3 normal_setting -1 /vepfs/wcf/G/zecheng/ckpt/lcao_chunk_16_size_1024/checkpoint-500
# bash scripts/inference_peft_model.sh ./longbench/llama-3_instruct /vepfs/wcf/G/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged llama-3 normal_setting -1 /vepfs/wcf/G/zecheng/ckpt/llama3_instruct_tuning/checkpoint-900

# bash scripts/inference_peft_model.sh ./longbench/lcao_v1-5 /data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged llama-3 normal_setting -1 /nvme/zecheng/ckpt/lcao_v1-5/checkpoint-100
# bash scripts/inference_peft_model.sh ./longbench/lcao_v6_ckpt_300 /vepfs/wcf/G/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged llama-3 normal_setting -1 /vepfs/wcf/G/zecheng/ckpt/lcao_chunk_16_size_1024/checkpoint-300
# bash scripts/inference_peft_model.sh ./longbench/llama_3_language_modeling /data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged llama-3 normal_setting -1 /nvme/zecheng/ckpt/llama3_post_train_langauge_modeling/checkpoint-100