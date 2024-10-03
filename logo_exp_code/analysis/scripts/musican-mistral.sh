CUDA_VISIBLE_DEVICES=0 python retrieval_head_detection.py --model_path "/vepfs/wcf/G/zecheng/hf_models/Mistral-7B-Instruct-v0.2" --s 0 --e 50000 --needle_file musican_girl.jsonl --needle_id 1 --model_name_suffix musican_girl > logs/musican-mistral-1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python retrieval_head_detection.py --model_path "/vepfs/wcf/G/zecheng/hf_models/Mistral-7B-Instruct-v0.2" --s 0 --e 50000 --needle_file musican_girl.jsonl --needle_id 2 --model_name_suffix musican_girl > logs/musican-mistral-2.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python retrieval_head_detection.py --model_path "/vepfs/wcf/G/zecheng/hf_models/Mistral-7B-Instruct-v0.2" --s 0 --e 50000 --needle_file musican_girl.jsonl --needle_id 3 --model_name_suffix musican_girl > logs/musican-mistral-3.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python retrieval_head_detection.py --model_path "/vepfs/wcf/G/zecheng/hf_models/Mistral-7B-Instruct-v0.2" --s 0 --e 50000 --needle_file musican_girl.jsonl --needle_id 4 --model_name_suffix musican_girl > logs/musican-mistral-4.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python retrieval_head_detection.py --model_path "/vepfs/wcf/G/zecheng/hf_models/Mistral-7B-Instruct-v0.2" --s 0 --e 50000 --needle_file musican_girl.jsonl --needle_id 5 --model_name_suffix musican_girl > logs/musican-mistral-5.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python retrieval_head_detection.py --model_path "/vepfs/wcf/G/zecheng/hf_models/Mistral-7B-Instruct-v0.2" --s 0 --e 50000 --needle_file musican_girl.jsonl --needle_id 6 --model_name_suffix musican_girl > logs/musican-mistral-6.log 2>&1 &

# CUDA_VISIBLE_DEVICES=6 python retrieval_head_detection.py --model_path "/vepfs/wcf/G/zecheng/hf_models/Mistral-7B-Instruct-v0.2" --s 0 --e 50000 --needle_file musican_girl.jsonl --needle_id 7 --model_name_suffix musican_girl > logs/musican-mistral-7.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python retrieval_head_detection.py --model_path "/vepfs/wcf/G/zecheng/hf_models/Mistral-7B-Instruct-v0.2" --s 0 --e 50000 --needle_file musican_girl.jsonl --needle_id 8 --model_name_suffix musican_girl > logs/musican-mistral-8.log 2>&1 &