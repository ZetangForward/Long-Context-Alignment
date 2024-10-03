CUDA_VISIBLE_DEVICES=0 python retrieval_head_detection_NIAH.py \
    --model_path /data/zecheng/hf_models/llama-2-7b-80k \
    --s 0 --e 50000 --needle_id 3 --model_name_suffix "NIAH_fix" \
    --needle_file "needles.jsonl" \
    --model_provider "LLaMA" > ./logs/llama2_niah-0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python retrieval_head_detection_NIAH.py \
    --model_path /data/zecheng/hf_models/llama-2-7b-80k \
    --s 0 --e 50000 --needle_id 4 --model_name_suffix "NIAH_fix" \
    --needle_file "needles.jsonl" \
    --model_provider "LLaMA" > ./logs/llama2_niah-1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python retrieval_head_detection_NIAH.py \
    --model_path /data/zecheng/hf_models/llama-2-7b-80k \
    --s 0 --e 50000 --needle_id 5 --model_name_suffix "NIAH_fix" \
    --needle_file "needles.jsonl" \
    --model_provider "LLaMA" > ./logs/llama2_niah-2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python retrieval_head_detection_NIAH.py \
    --model_path /data/zecheng/hf_models/llama-2-7b-80k \
    --s 0 --e 50000 --needle_id 6 --model_name_suffix "NIAH_fix" \
    --needle_file "needles.jsonl" \
    --model_provider "LLaMA" > ./logs/llama2_niah-3.log 2>&1 & 