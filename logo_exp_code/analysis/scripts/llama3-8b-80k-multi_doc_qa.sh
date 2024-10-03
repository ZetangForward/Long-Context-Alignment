CUDA_VISIBLE_DEVICES=0 python retrieval_head_detection_multi-hop.py \
    --model_path "/data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged" \
    --s 0 --e 50000 --needle_id 0 --model_name_suffix "multi_qa" \
    --needle_file "/data/zecheng/Retrieval_Head/haystack_for_detect/needle_multi_hop.jsonl" \
    --model_provider "llama3-8b-80k" 

