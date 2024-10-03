CUDA_VISIBLE_DEVICES=4 python retrieval_head_detection_multi-hop.py \
    --model_path /data/zecheng/hf_models/Qwen2-7B-Instruct \
    --s 0 --e 50000 --needle_id 0 --model_name_suffix "multi_qa" \
    --needle_file "/data/zecheng/Retrieval_Head/haystack_for_detect/needle_multi_hop.jsonl" \
    --model_provider "Qwen2-7B-Instruct" > logs/Qwen2-7B-Instruct-0.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python retrieval_head_detection_multi-hop.py \
    --model_path /data/zecheng/hf_models/Qwen2-7B-Instruct \
    --s 0 --e 50000 --needle_id 1 --model_name_suffix "multi_qa" \
    --needle_file "/data/zecheng/Retrieval_Head/haystack_for_detect/needle_multi_hop.jsonl" \
    --model_provider "Qwen2-7B-Instruct" > logs/Qwen2-7B-Instruct-1.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python retrieval_head_detection_multi-hop.py \
    --model_path /data/zecheng/hf_models/Qwen2-7B-Instruct \
    --s 0 --e 50000 --needle_id 2 --model_name_suffix "multi_qa" \
    --needle_file "/data/zecheng/Retrieval_Head/haystack_for_detect/needle_multi_hop.jsonl" \
    --model_provider "Qwen2-7B-Instruct" > logs/Qwen2-7B-Instruct-2.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python retrieval_head_detection_multi-hop.py \
    --model_path /data/zecheng/hf_models/Qwen2-7B-Instruct \
    --s 0 --e 50000 --needle_id 3 --model_name_suffix "multi_qa" \
    --needle_file "/data/zecheng/Retrieval_Head/haystack_for_detect/needle_multi_hop.jsonl" \
    --model_provider "Qwen2-7B-Instruct" > logs/Qwen2-7B-Instruct-3.log 2>&1 &