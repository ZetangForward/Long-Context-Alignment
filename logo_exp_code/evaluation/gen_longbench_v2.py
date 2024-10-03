from modelzipper.tutils import *
from fire import Fire
from utils import all_datasets, longbench_dataset_prompt, longbench_pred_length, load_model, DATASET2CATEGORY
import sys
from config.s1_80k_eval import EvalConfig

sys.path.append("/data/zecheng/sunzc/FlagEmbedding/Long_LLM/longllm_qlora")
from src import apply_chat_template

context_max_length = {"8k_setting": 7500, "tiny_setting": 15500, "normal_setting": 31500, "long_setting": 63500, "ultra_long_setting": 127500}
model_max_length = {"8k_setting": 8000, "tiny_setting": 16000, "normal_setting": 32000, "long_setting": 64000, "ultra_long_setting": 128000}


def create_position_ids(input_ids: torch.Tensor, max_context_length=65536, qa_size=1024):
    # input_ids: 1 x seq_len
    if input_ids.size(-1) < qa_size:
        return torch.arange(input_ids.size(-1)).unsqueeze(0)

    prefix_length = input_ids.size(-1) - qa_size // 2
    position_ids = list(range(prefix_length)) + list(range(max_context_length-qa_size, max_context_length - qa_size // 2))
    return torch.tensor(position_ids).long().unsqueeze(0)

def main(cfg: EvalConfig):
    
    if cfg.save_path is None:
        save_path = f"./longbench/{cfg.model_name}"
    else:
        save_path = cfg.save_path
    auto_mkdir(save_path)
    
    test_datasets = all_datasets
    
    dir_path = "./data"  # default setting
    
    print_c("quick check all file existing ...")

    for file_name in test_datasets:
        if not os.path.exists(os.path.join(dir_path, file_name + '.jsonl')):
            raise ValueError(f"{os.path.join(dir_path, file_name + '.jsonl')} does not exist...")
    
    max_context_length = context_max_length[cfg.model_max_length_setting]
    print_c(f"max_context_length are set as {max_context_length}")

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
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    test_model = load_model(cfg.model_path, cfg.peft_path, cfg.rope_theta, cfg.rope_factor, cfg.rope_type, cfg.max_position_embeddings, max_training_length=cfg.max_training_length)

    if hasattr(test_model, "generation_config"):
        eos_token_id = test_model.generation_config.eos_token_id
    else:
        eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
        
    eos_token_id.append(tokenizer.encode("\n", add_special_tokens=False)[-1])
    
    for cnt, dataset_name in enumerate(test_datasets):
        save_file_path = os.path.join(save_path, dataset_name + ".jsonl")
        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
        
        f = open(save_file_path, "a")
        file_path = os.path.join(dir_path, dataset_name + ".jsonl")
        
        datasets = auto_read_data(file_path)
        PROMPT_TEMPLATE, PRED_LENGTH = longbench_dataset_prompt[dataset_name], longbench_pred_length[dataset_name]
        with tqdm(total=len(datasets)) as pbar:
            for sample in datasets:
                if hasattr(test_model, "memory"):
                    test_model.memory.reset()

                context, input_, answers = sample['context'], sample['input'], sample['answers']
                prompt = PROMPT_TEMPLATE.format(input=input_, context=context)
                textual_input = tokenizer(prompt, return_tensors="pt").to(test_model.device).input_ids[0]

                if len(textual_input) > max_context_length:
                    half = int(max_context_length/2)
                    prompt = tokenizer.decode(textual_input[:half], skip_special_tokens=True) + \
                        tokenizer.decode(textual_input[-half:], skip_special_tokens=True)

                if not DATASET2CATEGORY[dataset_name] in ["EN Few-Shot Learning", "Code Completion"]:
                    prompt = apply_chat_template(
                        "llama-3", 
                        messages=[{'role': 'user', 'content': prompt}],
                        tokenizer=tokenizer,
                        add_generation_prompt=True,
                    ).raw

                input_ids = tokenizer(prompt, return_tensors="pt").to(test_model.device)
                position_ids = create_position_ids(input_ids.input_ids) # TODO: 这里将qa放在最后，和训练保持一致
                if dataset_name in ["2wikimqa_e", "hotpotqa_e", "musique", "multifieldqa_en", "qasper_e", "narrativeqa", "samsum_e"]:
                    outputs = test_model.generate(
                        **input_ids, 
                        position_ids=position_ids,
                        max_new_tokens=PRED_LENGTH, 
                        do_sample=None,
                        begin_suppress_tokens=eos_token_id,
                        eos_token_id=eos_token_id, temperature=None,
                        top_p=None,
                    )[0]
                else:
                    outputs = test_model.generate(
                        **input_ids,
                        position_ids=position_ids,
                        max_new_tokens=PRED_LENGTH,
                        num_beams=1,
                        min_new_tokens=1,
                        do_sample=None,
                        temperature=None,
                        top_p=None,
                    )[0]
                print(tokenizer.decode(outputs[input_ids.shape[-1]:], skip_special_tokens=False))
                pred_str = tokenizer.decode(outputs[input_ids.shape[-1]:], skip_special_tokens=True)
                saved_content = json.dumps({"dataset_name": dataset_name, "pred_str": pred_str, "answers": answers}) + '\n'
                pbar.update(1)
                f.write(saved_content)
                f.flush()
        f.close()
        print(f"process {cnt} files, remain {len(test_datasets) - cnt} files ...")


if __name__ == "__main__":
    cfg = EvalConfig()
    main(cfg)
    





