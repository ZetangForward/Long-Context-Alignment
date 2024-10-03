CUDA_VISIBLE_DEVICES=0 python retrieval_head_detection_multi-hop.py \
    --model_path /vepfs/wcf/G/zecheng/hf_models/llama-2-7b-80k \
    --s 0 --e 50000 --needle_id 0 --model_name_suffix "multi_qa" \
    --needle_file "/vepfs/wcf/G/zecheng/Retrieval_Head/haystack_for_detect/needle_multi_hop.jsonl" \
    --model_provider "LLaMA" > logs/multi_doc_qa-0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python retrieval_head_detection_multi-hop.py \
    --model_path /vepfs/wcf/G/zecheng/hf_models/llama-2-7b-80k \
    --s 0 --e 50000 --needle_id 1 --model_name_suffix "multi_qa" \
    --needle_file "/vepfs/wcf/G/zecheng/Retrieval_Head/haystack_for_detect/needle_multi_hop.jsonl" \
    --model_provider "LLaMA" > logs/multi_doc_qa-1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python retrieval_head_detection_multi-hop.py \
    --model_path /vepfs/wcf/G/zecheng/hf_models/llama-2-7b-80k \
    --s 0 --e 50000 --needle_id 2 --model_name_suffix "multi_qa" \
    --needle_file "/vepfs/wcf/G/zecheng/Retrieval_Head/haystack_for_detect/needle_multi_hop.jsonl" \
    --model_provider "LLaMA" > logs/multi_doc_qa-2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python retrieval_head_detection_multi-hop.py \
    --model_path /vepfs/wcf/G/zecheng/hf_models/llama-2-7b-80k \
    --s 0 --e 50000 --needle_id 3 --model_name_suffix "multi_qa" \
    --needle_file "/vepfs/wcf/G/zecheng/Retrieval_Head/haystack_for_detect/needle_multi_hop.jsonl" \
    --model_provider "LLaMA" > logs/multi_doc_qa-3.log 2>&1 &