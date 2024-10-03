SAVE_PATH="./longbench"
MODEL_NAME="llama3-8B-no-rope-type"

for LENGTH_TYPE in "normal_setting" "tiny_setting"
do
    echo "Running with LENGTH_TYPE=${LENGTH_TYPE}"

    CUDA_VISIBLE_DEVICES=0 python gen_longbench.py \
        -model_path="/home/export/base/ycsc_lijt1/lijt1/online1/zecheng/hf_models/Meta-Llama-3-8B-Instruct" \
        -peft_path="/home/export/base/ycsc_lijt1/lijt1/online1/zecheng/ckpt/s1-tmp2-no-rope-type" \
        -rope_theta=200e6 \
        -save_path=${SAVE_PATH}/${MODEL_NAME}-${LENGTH_TYPE} \
        -model_max_length_setting=${LENGTH_TYPE} \
        -model_name=${MODEL_NAME}

    echo "Generation for ${LENGTH_TYPE} finished, beginning to evaluate..."

    python eval_longbench.py --pred_path=${SAVE_PATH}/${MODEL_NAME}-${LENGTH_TYPE}

    echo "Evaluation for ${LENGTH_TYPE} finished."
done

echo "All settings processed."
