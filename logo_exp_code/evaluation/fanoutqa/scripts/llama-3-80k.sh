
CUDA_VISIBLE_DEVICES=7 python gen_fanout_qa.py \
    --model_path="/data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged" \
    --rope_theta 200e6 \
    --max_position_embeddings 65536 \
    --model_name "llama-3-80k" \

echo "generate finish ..."
