CUDA_VISIBLE_DEVICES=0 python retrieval_head_detection_multi-hop.py \
    --model_path /data/zecheng/hf_models/Mistral-7B-Instruct-v0.3 \
    --s 0 --e 50000 --needle_id 0 --model_name_suffix "multi_qa" \
    --needle_file "/data/zecheng/Retrieval_Head/haystack_for_detect/needle_multi_hop.jsonl" \
    --model_provider "Mistral-7B-Instruct-v0.3" > logs/Mistral-7B-Instruct-v0.3-0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python retrieval_head_detection_multi-hop.py \
    --model_path /data/zecheng/hf_models/Mistral-7B-Instruct-v0.3 \
    --s 0 --e 50000 --needle_id 1 --model_name_suffix "multi_qa" \
    --needle_file "/data/zecheng/Retrieval_Head/haystack_for_detect/needle_multi_hop.jsonl" \
    --model_provider "Mistral-7B-Instruct-v0.3" > logs/Mistral-7B-Instruct-v0.3-1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python retrieval_head_detection_multi-hop.py \
    --model_path /data/zecheng/hf_models/Mistral-7B-Instruct-v0.3 \
    --s 0 --e 50000 --needle_id 2 --model_name_suffix "multi_qa" \
    --needle_file "/data/zecheng/Retrieval_Head/haystack_for_detect/needle_multi_hop.jsonl" \
    --model_provider "Mistral-7B-Instruct-v0.3" > logs/Mistral-7B-Instruct-v0.3-2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python retrieval_head_detection_multi-hop.py \
    --model_path /data/zecheng/hf_models/Mistral-7B-Instruct-v0.3 \
    --s 0 --e 50000 --needle_id 3 --model_name_suffix "multi_qa" \
    --needle_file "/data/zecheng/Retrieval_Head/haystack_for_detect/needle_multi_hop.jsonl" \
    --model_provider "Mistral-7B-Instruct-v0.3" > logs/Mistral-7B-Instruct-v0.3-3.log 2>&1 &