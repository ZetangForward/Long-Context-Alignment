export CUDA_VISIBLE_DEVICES=1

python ./eval_needle.py \
    --max_length 64000 \
    --num_length_interval 15 \
    --min_length 8000 \
    --haystack_path "/vepfs/wcf/G/zecheng/Retrieval_Head/iclr2025/analysis/PaulGrahamEssays" \
    --result_dir ./results/Meta-Llama-3-8B-Instruct-Yarn \
    --rouge rouge-1 \
    --eva_indic p \
    --chat_template 'no' \
    --rope_type="dynamic" \
    --rope_factor=8.0 \
    --model_path /vepfs/wcf/G/zecheng/hf_models/Meta-Llama-3-8B-Instruct \
    




# result_dir = os.path.join(args.output_dir, args.result_dir)
# 如果打开load_result，将从result_dir/result.json中读取res

# you can use GPT3.5 as the scorer with the following command:
# export OPENAI_API_KEY="sk-xxxx"
# python -m main.eval_needle $COMMAND --min_length 8000 --max_length 80000 --enable_tp --gpt_eval

    # --rouge rouge-1 \
    # --eva_indic p \