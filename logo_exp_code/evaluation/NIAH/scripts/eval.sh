export CUDA_VISIBLE_DEVICES=$1
MODEL=$2
PEFT_PATH=$3
RESULT_DIR=$4

python ./eval_needle.py \
    --max_length 64000 \
    --num_length_interval 15 \
    --min_length 8000 \
    --haystack_path "/vepfs/wcf/G/zecheng/Retrieval_Head/iclr2025/analysis/PaulGrahamEssays" \
    --result_dir $RESULT_DIR \
    --rouge rouge-1 \
    --eva_indic p \
    --chat_template 'no' \
    --model_path $MODEL \
    --peft_path $PEFT_PATH \


# 下面是命令行
# bash scripts/eval.sh 0 /vepfs/wcf/G/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged "-1" ./results/Llama-3-8B-Instruct-80K-QLoRA-Merged
# bash scripts/eval.sh 2 /vepfs/wcf/G/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged /vepfs/wcf/G/zecheng/ckpt/llama3_instruct_tuning/checkpoint-1000 ./results/Llama-3-8B-80K-Instruct
# bash scripts/eval.sh 3 /vepfs/wcf/G/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged /vepfs/wcf/G/zecheng/ckpt/llama3_post_train_langauge_modeling/checkpoint-1000 ./results/Llama-3-8B-Instruct-80K-Language_modeling
# bash scripts/eval.sh 0  "-1" ./results/Meta-Llama-3-8B-Instruct-Yarn
# bash scripts/eval.sh 6 /vepfs/wcf/G/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged /vepfs/wcf/G/zecheng/ckpt/fix_lcao_chunk_16_size_1024/checkpoint-600/context_scaling ./results/v6
# bash scripts/eval.sh 7 /vepfs/wcf/G/zecheng/hf_models/Meta-Llama-3-8B-Instruct /vepfs/wcf/G/zecheng/ckpt/pose_chunk_size_1024/checkpoint-1000/context_scaling ./results/pose



# result_dir = os.path.join(args.output_dir, args.result_dir)
# 如果打开load_result，将从result_dir/result.json中读取res

# you can use GPT3.5 as the scorer with the following command:
# export OPENAI_API_KEY="sk-xxxx"
# python -m main.eval_needle $COMMAND --min_length 8000 --max_length 80000 --enable_tp --gpt_eval

    # --rouge rouge-1 \
    # --eva_indic p \