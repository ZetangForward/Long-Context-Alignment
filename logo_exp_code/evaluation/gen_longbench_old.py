from modelzipper.tutils import *
from fire import Fire
from utils import *
import sys

model_max_length = {"tiny_setting": 15000, "normal_setting": 31000, "long_setting": 63000, "ultra_long_setting": 127000}

def main(model_path: str = None, peft_path: str  = None, task_id: int = None, save_path: str = None, rope_theta: int = None, rope_factor: int = None, rope_type: int = None, model_name: str = None, max_position_embeddings: int = None, model_max_length_setting: str = "normal_setting"):

    if save_path is None:
        save_path = f"./longbench/{model_name}"
        auto_mkdir(save_path)

    if task_id is not None:
        test_datasets = [all_datasets[task_id]]
    else:
        test_datasets = all_datasets
    
    dir_path = "./data"  # default setting
    
    print_c("quick check all file existing ...")

    for file_name in test_datasets:
        if not os.path.exists(os.path.join(dir_path, file_name + '.jsonl')):
            raise ValueError(f"{os.path.join(dir_path, file_name + '.jsonl')} does not exist...")
    
    max_context_length = model_max_length[model_max_length_setting]
    print_c(f"max_context_length are set as {model_max_length}")

    if os.path.exists(save_path):
        already_finish_files = auto_read_dir(save_path, file_suffix=".jsonl")
        already_finish_files = [os.path.basename(f).split('.')[0] for f in already_finish_files]
        # check generated cases
        for f in already_finish_files:
            num_test_cases = len(auto_read_data(os.path.join(dir_path, f + ".jsonl")))
            num_pred_cases = len(auto_read_data(os.path.join(save_path, f + ".jsonl")))
            if num_test_cases != num_pred_cases: 
                print(f"{f} has not been processed, removing it from finished files ...")
                already_finish_files.remove(f)
    else:
        already_finish_files = []
        
    test_datasets = list(set(test_datasets) - set(already_finish_files))
    print(f"evaluating on datasets: {test_datasets}")

    # load models
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_model = load_old_model(model_path, peft_path, rope_theta, rope_factor, rope_type, max_position_embeddings)

    for cnt, dataset_name in enumerate(test_datasets):
        save_file_path = os.path.join(save_path, dataset_name + ".jsonl")
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        
        f = open(save_file_path, "a")
        file_path = os.path.join(dir_path, dataset_name + ".jsonl")
        
        datasets = auto_read_data(file_path)
        PROMPT_TEMPLATE, PRED_LENGTH = longbench_dataset_prompt[dataset_name], longbench_pred_length[dataset_name]
        with tqdm(total=len(datasets)) as pbar:
            for sample in datasets:
                context, input_, answers = sample['context'], sample['input'], sample['answers']
                prompt = PROMPT_TEMPLATE.format(input=input_, context=context)
                textual_input = tokenizer(prompt, return_tensors="pt").to(test_model.device).input_ids[0]

                if len(textual_input) > max_context_length:
                    half = int(max_context_length/2)
                    prompt = tokenizer.decode(textual_input[:half], skip_special_tokens=True) + \
                        tokenizer.decode(textual_input[-half:], skip_special_tokens=True)
                message = [
                        {"role": "user", "content": prompt},
                    ]
                input_ids = tokenizer(message, return_tensors="pt").input_ids.to(test_model.device)
                print(f"read_input_length is {input_ids.size(-1)}")
                outputs = test_model.generate(input_ids, use_cache=True, max_new_tokens=PRED_LENGTH, do_sample=False)
                pred_str = tokenizer.decode(outputs[0][input_ids.shape[-1]:])
                saved_content = json.dumps({"dataset_name": dataset_name, "pred_str": pred_str, "answers": answers}) + '\n'
                pbar.update(1)
                f.write(saved_content)
                f.flush()
        f.close()
        print(f"process {cnt} files, remain {len(test_datasets) - cnt} files ...")


if __name__ == "__main__":
    Fire(main)
    





