CUDA_VISIBLE_DEVICES=0 python retrieval_head_detection_NIAH.py \
    --model_path /data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged \
    --s 0 --e 50000 --needle_id 3 --model_name_suffix "NIAH_fix" \
    --needle_file "needles.jsonl" \
    --model_provider "LLaMA" > ./logs/llama3_niah-0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python retrieval_head_detection_NIAH.py \
    --model_path /data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged \
    --s 0 --e 50000 --needle_id 4 --model_name_suffix "NIAH_fix" \
    --needle_file "needles.jsonl" \
    --model_provider "LLaMA" > ./logs/llama3_niah-1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python retrieval_head_detection_NIAH.py \
    --model_path /data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged \
    --s 0 --e 50000 --needle_id 5 --model_name_suffix "NIAH_fix" \
    --needle_file "needles.jsonl" \
    --model_provider "LLaMA" > ./logs/llama3_niah-2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python retrieval_head_detection_NIAH.py \
    --model_path /data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged \
    --s 0 --e 50000 --needle_id 6 --model_name_suffix "NIAH_fix" \
    --needle_file "needles.jsonl" \
    --model_provider "LLaMA" > ./logs/llama3_niah-3.log 2>&1 & 

CUDA_VISIBLE_DEVICES=4 python retrieval_head_detection_NIAH.py \
    --model_path /data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged \
    --s 0 --e 50000 --needle_id 7 --model_name_suffix "NIAH_fix" \
    --needle_file "needles.jsonl" \
    --model_provider "LLaMA" > ./logs/llama3_niah-4.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python retrieval_head_detection_NIAH.py \
    --model_path /data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged \
    --s 0 --e 50000 --needle_id 8 --model_name_suffix "NIAH_fix" \
    --needle_file "needles.jsonl" \
    --model_provider "LLaMA" > ./logs/llama3_niah-5.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python retrieval_head_detection_NIAH.py \
    --model_path /data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged \
    --s 0 --e 50000 --needle_id 9 --model_name_suffix "NIAH_fix" \
    --needle_file "needles.jsonl" \
    --model_provider "LLaMA" > ./logs/llama3_niah-6.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python retrieval_head_detection_NIAH.py \
    --model_path /data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged \
    --s 0 --e 50000 --needle_id 10 --model_name_suffix "NIAH_fix" \
    --needle_file "needles.jsonl" \
    --model_provider "LLaMA" > ./logs/llama3_niah-7.log 2>&1 & 