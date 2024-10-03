model_path='/data/zecheng/hf_models/llama-2-7b-80k'
out_dir='/nvme/zecheng/data/iclr2025/llama2-train-data/long-llm-pred/chunk_16_size_1024'
input_data_dir='/nvme/zecheng/data/iclr2025/llama3-80k-train-data/long-llm-filtered/chunk_16_size_1024'

ulimit -n 4096
python gen_hf.py \
    --ports 4100 4101 4102 4103 4104 4105 4106 4107 \
    --tokenizer $model_path \
    --input_data_dir $input_data_dir \
    --output_dir $out_dir --K 1 \
    --max_workers 64 \
    --temperature 0.7 \
    --model_type 'llama-2' \
    --strategy 'half';