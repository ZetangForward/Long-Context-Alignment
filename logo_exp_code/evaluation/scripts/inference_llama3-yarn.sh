CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python gen_longbench.py \
    --model_path=/vepfs/wcf/G/zecheng/hf_models/Meta-Llama-3-8B-Instruct \
    --save_path=./longbench/llama3-8B-Instruct-YaRN-64K \
    --model_type=llama-3 \
    --rope_type="dynamic" \
    --rope_factor=8.0;

echo "generate finish ..., begin to evaluate ..."

python eval_longbench.py --pred_path=./longbench/llama3-8B-Instruct-YaRN-64K

