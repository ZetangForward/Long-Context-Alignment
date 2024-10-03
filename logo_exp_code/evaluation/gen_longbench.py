from modelzipper.tutils import *
from fire import Fire
from eval_utils import all_datasets, longbench_dataset_prompt, longbench_pred_length, load_model, DATASET2CATEGORY
import sys
from config.s1_80k_eval import EvalConfig
import argparse
from chat import apply_chat_template
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np

context_max_length = {"8k_setting": 7200, "tiny_setting": 15500, "normal_setting": 32000, "long_setting": 63500, "ultra_long_setting": 127500}
model_max_length = {"8k_setting": 8000, "tiny_setting": 16000, "normal_setting": 32000, "long_setting": 64000, "ultra_long_setting": 128000}

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_position_ids(input_ids: torch.Tensor, max_context_length=65536, qa_size=256):
    if input_ids.size(-1) < qa_size:
        return torch.arange(input_ids.size(-1)).unsqueeze(0)
    prefix_length = input_ids.size(-1) - qa_size
    position_ids = list(range(prefix_length)) + list(range(max_context_length-qa_size, max_context_length))
    return torch.tensor(position_ids, dtype=torch.LongTensor).unsqueeze(0)


def get_pred(rank: int = None, model_type: str=None, model_path: str = None, peft_path: str  = None, rope_theta: int = None, rope_factor: float = None, rope_type: int = None, max_position_embeddings: int = None, use_logn=False, max_testing_length=None, max_training_length: int = 8192, datasets= None, dataset_name=None, max_context_length: int=None, return_list=None):
    
    print(f"rank {rank} is processing {dataset_name} length {len(datasets)} ...")
    # load models
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_model = load_model(model_type, model_path, peft_path, rope_theta, rope_factor, rope_type, max_position_embeddings, max_testing_length=max_testing_length, max_training_length=max_training_length, use_logn=use_logn, device=torch.device(f'cuda:{rank}'))

    if hasattr(test_model, "generation_config"):
        eos_token_id = test_model.generation_config.eos_token_id
    else:
        eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
        
    eos_token_id.append(tokenizer.encode("\n", add_special_tokens=False)[-1])
    
    PROMPT_TEMPLATE, PRED_LENGTH = longbench_dataset_prompt[dataset_name], longbench_pred_length[dataset_name]
    pred_res = []
    with torch.no_grad(), tqdm(total=len(datasets)) as pbar:
        for sample in datasets:
            if hasattr(test_model, "memory"):
                test_model.memory.reset()

            context, input_, answers = sample['context'], sample['input'], sample['answers']
            prompt = PROMPT_TEMPLATE.format(input=input_, context=context)

            if not DATASET2CATEGORY[dataset_name] in ["EN Few-Shot Learning", "Code Completion"]:
                prompt = apply_chat_template(
                    model_type, 
                    messages=[{'role': 'user', 'content': prompt}],
                    tokenizer=tokenizer,
                    add_generation_prompt=True,
                ).raw

            textual_input = tokenizer(prompt, return_tensors="pt").to(test_model.device).input_ids[0]

            max_context_length = max_context_length - PRED_LENGTH - 100 # for chat template
            if len(textual_input) > max_context_length:
                half = int(max_context_length/2)
                prompt = tokenizer.decode(textual_input[:half], skip_special_tokens=True) + \
                    tokenizer.decode(textual_input[-half:], skip_special_tokens=True)

            input_ids = tokenizer(prompt, return_tensors="pt").to(test_model.device).input_ids

            if dataset_name in ["2wikimqa_e", "hotpotqa_e", "musique", "multifieldqa_en", "qasper_e", "narrativeqa", "samsum_e"]:
                outputs = test_model.generate(
                    input_ids, 
                    max_new_tokens=PRED_LENGTH, 
                    do_sample=None,
                    begin_suppress_tokens=eos_token_id,
                    eos_token_id=eos_token_id, temperature=None,
                    top_p=None,
                )[0]
            elif dataset_name in ['gov_report_e', 'qmsum_e', 'multi_news_e']:
                outputs = test_model.generate(
                    input_ids,
                    max_new_tokens=PRED_LENGTH,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]
            else:
                outputs = test_model.generate(
                    input_ids,
                    max_new_tokens=PRED_LENGTH,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]

            pred_str = tokenizer.decode(outputs[input_ids.shape[-1]:], skip_special_tokens=True)
            pred_res.append({"dataset_name": dataset_name, "pred_str": pred_str, "answers": answers}) 
            pbar.update(1)
            
    return_list.extend(pred_res)
    
    return 'finished'



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lb testing")
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model')
    parser.add_argument('--peft_path', type=str, default=None, help='Path to the PEFT model')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the output')
    parser.add_argument('--rope_theta', type=float, default=None, help='RoPE theta value')
    parser.add_argument('--rope_factor', type=float, default=None, help='RoPE factor')
    parser.add_argument('--rope_type', type=str, default=None, help='RoPE type')
    parser.add_argument('--model_type', type=str, default=None, help='Name of the model')
    parser.add_argument('--max_position_embeddings', type=int, default=None, help='Maximum position embeddings')
    parser.add_argument('--model_max_length_setting', type=str, default="normal_setting", help='Model max length setting')
    parser.add_argument('--max_training_length', type=int, default=8192, help='Maximum training length')
    parser.add_argument('--seed', type=int, default=27, help='default seed')
    parser.add_argument('--use_logn', action='store_true', help='use logn')

    args = parser.parse_args()
    if args.max_position_embeddings == -1:
        args.max_position_embeddings = None
    if args.rope_theta == -1:
        args.rope_theta = None
        
    print_c(args)
    
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)
    
    log_c(f'begin to eval on {world_size} gpus ...')
    
    if args.save_path is None:
        save_path = f"./longbench/{args.model_type}"
    auto_mkdir(args.save_path)
    
    dir_path = "./data"  # default setting
    
    print_c("quick check all file existing ...")

    for file_name in all_datasets:
        if not os.path.exists(os.path.join(dir_path, file_name + '.jsonl')):
            raise ValueError(f"{os.path.join(dir_path, file_name + '.jsonl')} does not exist...")
    
    max_context_length = context_max_length[args.model_max_length_setting]
    print_c(f"max_context_length are set as {max_context_length}")

    if os.path.exists(args.save_path):
        already_finish_files = auto_read_dir(args.save_path, file_suffix=".jsonl")
        already_finish_files = [os.path.basename(f).split('.')[0] for f in already_finish_files]
        
        # check generated cases
        for f in already_finish_files:
            num_test_cases = len(auto_read_data(os.path.join(dir_path, f + ".jsonl")))
            num_pred_cases = len(auto_read_data(os.path.join(args.save_path, f + ".jsonl")))
            if num_test_cases != num_pred_cases: 
                print(f"{f} has not been processed, removing it from finished files ...")
                already_finish_files.remove(f)
    else:
        already_finish_files = []
        
    test_datasets = list(set(all_datasets) - set(already_finish_files))
    print(f"evaluating on datasets: {test_datasets}")
    torch.cuda.manual_seed_all(args.seed)

    for dataset_name in test_datasets:
        test_data = auto_read_data(os.path.join(dir_path, dataset_name + ".jsonl"))
        
        save_res_path = os.path.join(args.save_path, dataset_name + ".jsonl")
        # os.makedirs(os.path.dirname(save_res_path), exist_ok=True)
        
        data_subsets = [test_data[i::world_size] for i in range(world_size)]

        with tqdm(total=world_size) as pbar:
            def update(*args):
                pbar.update()
            
            processes = []
            manager = mp.Manager()
            return_list = manager.list()
             
            for rank in range(world_size):
                p = mp.Process(target=get_pred, args=(rank, args.model_type, args.model_path, args.peft_path, args.rope_theta, args.rope_factor, args.rope_type, args.max_position_embeddings, args.use_logn, max_context_length, args.max_training_length, data_subsets[rank], dataset_name, max_context_length, return_list))
                p.start()
                processes.append(p)
            
            for p in processes:
                p.join()
                update()
        
        results = list(return_list)

        auto_save_data(results, save_res_path)