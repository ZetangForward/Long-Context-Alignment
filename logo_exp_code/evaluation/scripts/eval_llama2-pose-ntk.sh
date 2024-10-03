SAVE_PATH="./longbench"
MODEL_NAME="LLaMA2-7B-PoSE-NTK-16k"

CUDA_VISIBLE_DEVICES=6 python gen_longbench.py \
    --model_path="/data/zecheng/hf_models/LLaMA2-7B-PoSE-NTK-16k" \
    --save_path=${SAVE_PATH}/${MODEL_NAME} \
    --model_name=${MODEL_NAME} \
    --rope_type="dynamic" --rope_factor=4.0 \
    --model_max_length_setting="tiny_setting";

echo "generate finish ..., begin to evaluate ..."

python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}
