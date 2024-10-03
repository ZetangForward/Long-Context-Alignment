CUDA_VISIBLE_DEVICES=0 python retrieval_head_detection_multi-hop.py \
    --model_path "/data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged" \
    --s 0 --e 50000 --needle_id 0 --model_name_suffix "multi_qa" \
    --needle_file "/data/zecheng/Retrieval_Head/haystack_for_detect/needle_multi_hop.jsonl" \
    --model_provider "llama3-8b-80k" > logs/llama3-8b-80k_0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python retrieval_head_detection_multi-hop-llama3.py \
    --model_path /data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged \
    --s 0 --e 50000 --needle_id 1 --model_name_suffix "multi_qa" \
    --needle_file "/data/zecheng/Retrieval_Head/haystack_for_detect/needle_multi_hop.jsonl" \
    --model_provider "llama3-8b-80k" > logs/llama3-8b-80k_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python retrieval_head_detection_multi-hop-llama3.py \
    --model_path /data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged \
    --s 0 --e 50000 --needle_id 2 --model_name_suffix "multi_qa" \
    --needle_file "/data/zecheng/Retrieval_Head/haystack_for_detect/needle_multi_hop.jsonl" \
    --model_provider "llama3-8b-80k" > logs/llama3-8b-80k_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python retrieval_head_detection_multi-hop-llama3.py \
    --model_path /data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged \
    --s 0 --e 50000 --needle_id 3 --model_name_suffix "multi_qa" \
    --needle_file "/data/zecheng/Retrieval_Head/haystack_for_detect/needle_multi_hop.jsonl" \
    --model_provider "llama3-8b-80k" > logs/llama3-8b-80k_3.log 2>&1 &