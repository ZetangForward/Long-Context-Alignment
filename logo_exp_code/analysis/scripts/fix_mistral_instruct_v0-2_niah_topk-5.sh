CUDA_VISIBLE_DEVICES=0 python retrieval_head_detection_NIAH.py \
    --model_path /data/zecheng/hf_models/Mistral-7B-Instruct-v0.2 \
    --s 0 --e 50000 --needle_id 7 --model_name_suffix "NIAH_fix_topk-5" --topk 5 \
    --needle_file "needles.jsonl" > ./logs/mistral_niah-7.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python retrieval_head_detection_NIAH.py \
    --model_path /data/zecheng/hf_models/Mistral-7B-Instruct-v0.2 \
    --s 0 --e 50000 --needle_id 8 --model_name_suffix "NIAH_fix_topk-5" --topk 5 \
    --needle_file "needles.jsonl" > ./logs/mistral_niah-8.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python retrieval_head_detection_NIAH.py \
    --model_path /data/zecheng/hf_models/Mistral-7B-Instruct-v0.2 \
    --s 0 --e 50000 --needle_id 9 --model_name_suffix "NIAH_fix_topk-5" --topk 5 \
    --needle_file "needles.jsonl" > ./logs/mistral_niah-9.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python retrieval_head_detection_NIAH.py \
    --model_path /data/zecheng/hf_models/Mistral-7B-Instruct-v0.2 \
    --s 0 --e 50000 --needle_id 10 --model_name_suffix "NIAH_fix_topk-5" --topk 5 \
    --needle_file "needles.jsonl" > ./logs/mistral_niah-10.log 2>&1 & 