CUDA_VISIBLE_DEVICES=0 python retrieval_head_detection.py --model_path /vepfs/wcf/G/zecheng/hf_models/Llama-2-7b-chat --s 0 --e 50000 --needle_file /vepfs/wcf/G/zecheng/Retrieval_Head/haystack_for_detect/musican_girl.jsonl --test_cases 1 --model_name_suffix musican_girl > logs/musican-llama-1.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python retrieval_head_detection.py --model_path /vepfs/wcf/G/zecheng/hf_models/Llama-2-7b-chat --s 0 --e 50000 --needle_file /vepfs/wcf/G/zecheng/Retrieval_Head/haystack_for_detect/musican_girl.jsonl --test_cases 2 --model_name_suffix musican_girl > logs/musican-llama-2.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python retrieval_head_detection.py --model_path /vepfs/wcf/G/zecheng/hf_models/Llama-2-7b-chat --s 0 --e 50000 --needle_file /vepfs/wcf/G/zecheng/Retrieval_Head/haystack_for_detect/musican_girl.jsonl --test_cases 3 --model_name_suffix musican_girl > logs/musican-llama-3.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python retrieval_head_detection.py --model_path /vepfs/wcf/G/zecheng/hf_models/Llama-2-7b-chat --s 0 --e 50000 --needle_file /vepfs/wcf/G/zecheng/Retrieval_Head/haystack_for_detect/musican_girl.jsonl --test_cases 4 --model_name_suffix musican_girl > logs/musican-llama-4.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 python retrieval_head_detection.py --model_path /vepfs/wcf/G/zecheng/hf_models/Llama-2-7b-chat --s 0 --e 50000 --needle_file /vepfs/wcf/G/zecheng/Retrieval_Head/haystack_for_detect/musican_girl.jsonl --test_cases 5 --model_name_suffix musican_girl > logs/musican-llama-5.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python retrieval_head_detection.py --model_path /vepfs/wcf/G/zecheng/hf_models/Llama-2-7b-chat --s 0 --e 50000 --needle_file /vepfs/wcf/G/zecheng/Retrieval_Head/haystack_for_detect/musican_girl.jsonl --test_cases 6 --model_name_suffix musican_girl > logs/musican-llama-6.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python retrieval_head_detection.py --model_path /vepfs/wcf/G/zecheng/hf_models/Llama-2-7b-chat --s 0 --e 50000 --needle_file /vepfs/wcf/G/zecheng/Retrieval_Head/haystack_for_detect/musican_girl.jsonl --test_cases 7 --model_name_suffix musican_girl > logs/musican-llama-7.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 python retrieval_head_detection.py --model_path /vepfs/wcf/G/zecheng/hf_models/Llama-2-7b-chat --s 0 --e 50000 --needle_file /vepfs/wcf/G/zecheng/Retrieval_Head/haystack_for_detect/musican_girl.jsonl --test_cases 8 --model_name_suffix musican_girl > logs/musican-llama-8.log 2>&1 &