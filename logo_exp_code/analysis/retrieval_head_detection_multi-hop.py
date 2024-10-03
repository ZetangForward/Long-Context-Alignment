
#import tiktoken
import os 
import glob
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
import sys
sys.path.append("./faiss_attn/")
from source.modeling_llama import LlamaForCausalLM
from source.modeling_qwen2 import Qwen2ForCausalLM
from source.modeling_mixtral import MixtralForCausalLM
from source.modeling_mistral import MistralForCausalLM
from source.modeling_self_extend_llama import SelfExtendLlamaForCausalLM
from llama_rope_scaled_monkey_patch import replace_llama_with_condense
import numpy as np
import argparse
from rouge_score import rouge_scorer
from datetime import datetime, timezone
from collections import defaultdict
import time
import torch
from modelzipper.tutils import *
import yarn.configuration_llama as yarn_llama_cfg
import yarn.modeling_llama_yarn as yarn_llama
import itertools
from eval_utils import CustomScorer


def reset_rope(model, model_max_train_len, scaling_factor):
    for l in model.model.layers:
        l.self_attn.rotary_emb.scaling_factor = scaling_factor
        l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=model_max_train_len, device=l.self_attn.rotary_emb.inv_freq.device, dtype=torch.float32)


class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(
        self, haystack_dir="./haystack_for_detect", context_lengths_min = 1000, context_lengths_max = 50000, 
        context_lengths_num_intervals = 20, context_lengths = None, document_depth_percent_min = 0, document_depth_percent_max = 100, 
        document_depth_percent_intervals = 10, document_depth_percents = None, document_depth_percent_interval_type = "linear",
        model_provider = "OpenAI", model_name='', model_name_suffix=None, num_concurrent_requests = 1, save_results = True,
        save_contexts = False, final_context_length_buffer = 200, seconds_to_sleep_between_completions = None, print_ongoing_status = True,
        needle_file = None, needle_id = 0, peft_model_path = None
    ):
        
        needles_and_stacks = auto_read_data(needle_file)
        # self.shortcut_keys = auto_read_data(f"{haystack_dir}/shortcut_key.jsonl")
        self.needle_list = [(l["needle1"], l["needle2"]) for l in needles_and_stacks]
        self.haystack_dir_list = [f"{haystack_dir}/part{i}" for i in range(1, 4)]
        self.retrieval_question_list = [l["question"] for l in needles_and_stacks]
        self.real_ansers_list = [l["real_needle"] for l in needles_and_stacks]
        self.forbidden_string_lst = [l["forbidden_strings"] for l in needles_and_stacks]

        self.peft_model_path = peft_model_path
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider

        self.testing_results = []
        self.head_counter = defaultdict(list)
        self.reference_counter = defaultdict(list)
        self.instance_counter = defaultdict(list)
        
        if("/" in model_name): self.model_version = model_name.split("/")[-1]
        else: self.model_version = model_name
        
        if(model_name_suffix is not None): self.model_version += "_" + model_name_suffix

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
            raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
        else:
            if document_depth_percent_interval_type == 'linear':
                self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
            elif document_depth_percent_interval_type == 'sigmoid':
                self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
   
        self.model_name = model_name
        self.enc = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print("loading from %s" % model_name)
        self.load_model()
            
        if "CUDA_VISIBLE_DEVICES" in os.environ: self.multi_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"]) > 1
        else: self.multi_gpus = True
        
        self.needle_id = needle_id
        self.haystack_dir = self.haystack_dir_list[0]
        self.needle = self.needle_list[needle_id]  # (needle1, needle2)
        if "qwen2" in self.model_provider.lower():
            self.needle_tok1, self.needle_tok2 = self.enc(self.needle[0]).input_ids, self.enc(self.needle[1]).input_ids
        else:
            self.needle_tok1, self.needle_tok2 = self.enc(self.needle[0]).input_ids[1:], self.enc(self.needle[1]).input_ids[1:]
        self.forbidden_strings = self.forbidden_string_lst[needle_id]
        self.real_answer = self.real_ansers_list[needle_id]
        self.real_answer_tok = self.enc(self.real_answer).input_ids[1:]
        self.retrieval_question = self.retrieval_question_list[needle_id]
       
        self.model_to_test_description = model_name
        model_name = model_name.split('/')[-1]

        self.scorer = CustomScorer(self.real_answer, self.forbidden_strings)
    
    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    

    def load_model(self):
        if 'yarn' in self.model_version:
            config = yarn_llama_cfg.LlamaConfig.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
        self.layer_num, self.head_num = config.num_hidden_layers, config.num_attention_heads
        print(f"layer number: {self.layer_num}, head number {self.head_num}")
        if "Qwen" in self.model_version:
            self.model_to_test = Qwen2ForCausalLM.from_pretrained(model_name,torch_dtype="auto",device_map='auto',use_flash_attention_2="flash_attention_2").eval()
        elif "Mixtral" in self.model_version:
            self.model_to_test = MixtralForCausalLM.from_pretrained(model_name,torch_dtype="auto",device_map='auto',use_flash_attention_2="flash_attention_2", trust_remote_code=True).eval()
        elif "Mistral" in self.model_version or "FILM" in self.model_version:
            self.model_to_test = MistralForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,device_map='auto',use_flash_attention_2="flash_attention_2",trust_remote_code=True).eval()
        elif 'llama3-8b-80k' in self.model_provider.lower():
            print("loading from llama-3-8b-80k successfully ...")
            self.model_to_test = LlamaForCausalLM.from_pretrained(
                model_name, use_flash_attention_2="flash_attention_2", torch_dtype="auto", device_map='auto').eval()
        elif 'llama-2-7b-80k' in self.model_version:
            print("loading from llama-2-7b-80k successfully ...")
            self.model_to_test = LlamaForCausalLM.from_pretrained(model_name,use_flash_attention_2="flash_attention_2",torch_dtype="auto",device_map='auto').eval()
            scaling_factor = 10
            reset_rope(self.model_to_test, model_max_train_len=81920, scaling_factor=scaling_factor)
        elif 'yarn' in self.model_version:
            self.model_to_test = yarn_llama.LlamaForCausalLM.from_pretrained(model_name, config=config, torch_dtype="auto",device_map='auto',use_flash_attention_2="flash_attention_2")
        elif self.model_version.lower() == 'Llama-2-7b-chat'.lower():
            replace_llama_with_condense(2, 4)
            self.model_to_test = LlamaForCausalLM.from_pretrained(model_name,use_flash_attention_2="flash_attention_2", torch_dtype=torch.bfloat16,device_map='auto').eval()
        elif 'SelfExtend' in self.model_version:
            self.model_to_test = SelfExtendLlamaForCausalLM.from_pretrained(model_name, use_flash_attention_2="flash_attention_2", torch_dtype=torch.bfloat16, device_map='auto').eval()
        elif "chatglm3-6b-128k" in self.model_provider.lower():
            self.model_to_test = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto').eval()
        elif 'Llama-2-7b-hf' in self.model_version:
            print("loading from %s successfully ..." % self.model_version)
            config = AutoConfig.from_pretrained(model_name)
            config.rope_scaling = {"type": "dynamic", "factor": 10}
            config.rope_theta = 1e5
            config.max_position_embeddings = 40960
            self.model_to_test = LlamaForCausalLM.from_pretrained(model_name, config=config, use_flash_attention_2="flash_attention_2", torch_dtype=torch.bfloat16, device_map='auto').eval()
        else:
            self.model_to_test = AutoModelForCausalLM.from_pretrained(model_name,use_flash_attention_2="flash_attention_2", torch_dtype=torch.bfloat16,device_map='auto').eval()


    def run_test(self, args): # Run through each iteration of context_lengths and depths
        for context_length in self.context_lengths:
            if context_length < args.s_len or context_length > args.e_len: continue
            combinations_depth_percents = list(itertools.combinations(self.document_depth_percents, 2))  # 2 hop questions
            # copy the self.document_depth_percents for twice insert
            for depth_percents in combinations_depth_percents:  # depth 1 and depth 2
                _ = self.evaluate_and_log(context_length, depth_percents)


    def retrieval_calculate(self, attention_maxtrix, attribution_scores, inp, step_token, citation_scores, step_i, max_record_len, topk=1):
        (ref_st1, ref_ed1), (ref_st2, ref_ed2) = self.reference_pos[0], self.reference_pos[1]
        (needle_st1, needle_ed1), (needle_st2, needle_ed2) = self.needle_pos[0], self.needle_pos[1]
        
        for layer_idx in range(self.layer_num):
            for head_idx in range(self.head_num):
                values, idx = attention_maxtrix[layer_idx][0][head_idx][-1].topk(topk)
                # citation scores
                if step_i < max_record_len:  # only record the max_record_len steps
                    for v, i in zip(values, idx):  
                        if (ref_st1 <= i < ref_ed1) or (ref_st2 <= i < ref_ed2):  
                            citation_scores[layer_idx][head_idx][step_i].append((i.item(), v.item())) 
                        else:
                            citation_scores[layer_idx][head_idx][step_i].append((i.item(), 0))
                # needle scores (attribution scores)
                for v, i in zip(values, idx):  # find the top 1 attention id for the needle hit
                    is_i_in_needle_pos = (needle_st1 <= i < needle_ed1) and (needle_st2 <= i < needle_ed2) 
                    if is_i_in_needle_pos and inp.item() == self.prompt_ids[i].item():
                        attribution_scores[layer_idx][head_idx][0] += 1/(needle_ed1-needle_st1)
                        attribution_scores[layer_idx][head_idx][1] += step_token
                        break
    

    def citation_head_accumulate(self, reference_score, ini_recall_score, pen_recall_score=0):
        for layer_idx in range(self.layer_num):
            for head_idx in range(self.head_num): # per head per layer save the reference score
                self.reference_counter[f"{layer_idx}-{head_idx}"].append(
                    (reference_score[layer_idx][head_idx], ini_recall_score, pen_recall_score)
                ) 
    

    def attribute_head_accumulate(self, retrieval_score, ini_recall_score, pen_recall_score=0):
        for layer_idx in range(self.layer_num):
            for head_idx in range(self.head_num):
                self.head_counter[f"{layer_idx}-{head_idx}"].append(
                    (retrieval_score[layer_idx][head_idx][0], ini_recall_score, pen_recall_score)
                )
                
                
    def instance_score_accumulate(self, hit_candidate_scores, entropy_scores, context_length, depth_percent):
        self.instance_counter[f'{context_length}-{depth_percent}'].append((hit_candidate_scores, entropy_scores))


    def decode(self, q_outputs, inp, decode_len):
        ## lens inner the model
        attribution_scores = [[[0, ''] for _ in range(self.head_num)] for _ in range(self.layer_num)]
        citation_scores = [[[[] for _ in range(decode_len)] for _ in range(self.head_num)] for _ in range(self.layer_num)]
        
        ## model output
        # hit_candidate_scores = [[] for _ in range(decode_len)]
        gen_candidates, output = [], []

        needle_entropy = 0.0
        past_kv = q_outputs.past_key_values
        
        for step_i in range(decode_len):
            inp = inp.view(1, 1)
            outputs = self.model_to_test(input_ids=inp, past_key_values=past_kv, use_cache=True, output_attentions=True, attn_mode="torch")
            past_kv = outputs.past_key_values
            # probabilities = torch.functional.F.softmax(outputs.logits, dim=-1)
            # needle_entropy += -probabilities[:,:,self.needle_tok[step_i]] * torch.log(probabilities[:,:,self.needle_tok[step_i]] + 1e-9).item()
            topk_values, topk_indices = torch.topk(outputs.logits[0, -1], 5, dim=-1)
            gen_candidates.append((topk_indices, topk_values))
            inp = outputs.logits[0, -1].argmax()
            step_token = self.enc.convert_ids_to_tokens(inp.item())
            output.append(inp.item())

            self.retrieval_calculate(
                attention_maxtrix=outputs.attentions, 
                attribution_scores=attribution_scores, 
                inp=inp, 
                step_token=step_token, 
                citation_scores=citation_scores, 
                step_i=step_i, 
                max_record_len=decode_len, 
                topk=1
            )
        
        # search for the generation path
        # for step_i, (topk_indices, topk_values) in enumerate(gen_candidates):
        #     if self.real_answer_tok[step_i] in topk_indices:  # record the hit candidate
        #         rk_index = torch.where(topk_indices == self.real_answer_tok[step_i])[0].item()
        #         hit_candidate_scores[step_i].append((rk_index, round(topk_values[rk_index].item(), 3)))

        return {"model_generation": output, "attribution_scores": attribution_scores, "hit_candidate_scores": gen_candidates,
                "citation_scores": citation_scores, "needle_entropy": needle_entropy}


    def find_needle_idx(self, needle):
        if "llama3" in self.model_provider.lower() or "qwen2" in self.model_provider.lower():
            needle = " " + needle
        needle_pos = []
        needle_ids = self.enc(needle, add_special_tokens=False)["input_ids"]
        print(self.enc.decode(needle_ids, skip_special_tokens=False))
        span_len = len(needle_ids)
        i = 0
        while i < len(self.prompt_ids):            
            token_span = self.prompt_ids[i : i + span_len]
            span_ids = set(token_span.tolist())
            overlap = float(len(span_ids.intersection(set(needle_ids)))) / len(set(needle_ids))
            if(overlap > 0.9): 
                needle_pos.append((i, i + span_len))
                i += span_len
            else: 
                i += 1
        if len(needle_pos) != 2:
            print(needle_pos)
            import pdb; pdb.set_trace()
        return needle_pos
    

    def find_citation_idx(self, reference):
        citation_pos = []  # reference = (needle_1, needle_2)
        for needle in reference:
            needle_ids = self.enc(needle, add_special_tokens=False)["input_ids"]
            print(self.enc.decode(needle_ids, skip_special_tokens=False))
            span_len = len(needle_ids)
            i = 0
            while i < len(self.prompt_ids):            
                token_span = self.prompt_ids[i : i + span_len]
                span_ids = set(token_span.tolist())
                overlap = float(len(span_ids.intersection(set(needle_ids)))) / len(set(needle_ids))
                if(overlap > 0.9): 
                    citation_pos.append((i, i + span_len))
                    i += span_len
                else:
                    i += 1
        if len(citation_pos) != 2:
            print(citation_pos)
            import pdb; pdb.set_trace()
        return citation_pos  # [(pos1, pos2), (pos1, pos2)]
    
    
    def evaluate_and_log(self, context_length, depth_percent):
        context, insert_meta_data = self.generate_context(context_length, depth_percent)
        question = f"\nQuestion: {self.retrieval_question}"
        if self.model_provider in ["Mistral-7B-Instruct-v0.3", "Qwen1.5-14B-Chat"]:
            prompt = [{"role": "user", "content": f"<book> {context} </book>\nBased on the content of the book, Question: {self.retrieval_question} Just provide the answer without explanation.\nAnswer:"},]
            input_ids = self.enc.apply_chat_template(conversation=prompt, tokenize=True, add_generation_prompt=True, return_tensors='pt')
            max_new_tokens = 64
        elif self.model_provider in ["Qwen2-7B-Instruct"]:
            print(f"apply {self.model_version} chat template ...")
            prompt = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"<book> {context} </book>\nBased on the content of the book, Question: {self.retrieval_question}. Just provide the answer without explanation.\nAnswer:"}
            ]
            input_ids = self.enc.apply_chat_template(conversation=prompt, tokenize=True, add_generation_prompt=True, return_tensors='pt')
            max_new_tokens = 64
            print(f"model generation length is {max_new_tokens}")
        elif self.model_provider.lower() == "llama3-8b-80k":
            messages = [{"role": "user", "content": f"<book> {context} </book>\nBased on the content of the book, Question: {self.retrieval_question}. Just provide the answer without explanation.\nAnswer:"}]
            input_ids = self.enc.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            max_new_tokens = 32
        else:
            input_context = context + question
            input_ids = self.enc(input_context , return_tensors="pt")['input_ids']
            max_new_tokens = 32
        
        # Prepare your message to send to the model you're going to evaluate
        test_start_time = time.time()
        self.prompt_ids = input_ids[0, :]
        if not self.multi_gpus:
            input_ids = input_ids.to(self.model_to_test.device)
        self.needle_pos = self.find_needle_idx(self.real_answer)
        self.reference_pos = self.find_citation_idx(self.needle)
        
        if len(self.reference_pos) != 2 or len(self.needle_pos) != 2:
            print(f"skip {context_length} {depth_percent}")
            return None
        
        with torch.no_grad():
            q_outputs = self.model_to_test(input_ids=input_ids[:,:-1], use_cache=True, return_dict=True)
            decode_res = self.decode(q_outputs, input_ids[:,-1], len(self.real_answer_tok) + max_new_tokens)
            response = self.enc.decode(decode_res["model_generation"], skip_special_tokens=True).strip()

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        initial_recall_score, penality_recall_score = self.scorer.score(response.split('\n')[0])
        
        self.citation_head_accumulate(decode_res['citation_scores'], initial_recall_score, penality_recall_score)
        self.attribute_head_accumulate(decode_res['attribution_scores'], initial_recall_score, penality_recall_score)
        self.instance_score_accumulate(decode_res['hit_candidate_scores'], decode_res['needle_entropy'], context_length, depth_percent)
        
        head_score = [(i[0], np.mean(i[1])) for i in self.head_counter.items()]
        head_score = sorted(head_score, key=lambda x: x[1], reverse=True)
        print([[i[0]] for i in head_score][:15])

        results = {
            'model' : self.model_to_test_description,
            'context_length' : int(context_length),
            'depth_percent' : "_".join([str(i) for i in depth_percent]),
            'version' : self.results_version,
            'needle' : self.needle,
            'question': self.retrieval_question,
            'model_response' : response,
            'score' : initial_recall_score,
            'penalized_recall_score' : penality_recall_score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"ini_rouge_score: {initial_recall_score}")
            print (f"pen_rouge_score: {penality_recall_score}")
            print (f"Response: {response}\n")

        context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{"_".join([str(t) for t in depth_percent])}'

        if self.save_contexts:
            results['file_name'] = context_file_location
            # Save the context to file for retesting
            if not os.path.exists('contexts'):
                os.makedirs('contexts')
            if not os.path.exists(f'contexts/{self.model_version}_case{self.needle_id}'):
                os.makedirs(f'contexts/{self.model_version}_case{self.needle_id}')
            with open(f'contexts/{self.model_version}_case{self.needle_id}/{context_file_location}_context.txt', 'w') as f:
                f.write(context)
            
        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists(f'results/graph/{self.model_version}_case{self.needle_id}'):
                os.makedirs(f'results/graph/{self.model_version}_case{self.needle_id}')
            # Save the result to file for retesting
            p = f'results/graph/{self.model_version}_case{self.needle_id}/{context_file_location}_results.json'
            print("Writing at %s" % p)
            with open(p, 'w') as f:
                json.dump(results, f)


    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """
        results_dir = 'results/' + self.model_version
        print("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    # import ipdb; ipdb.set_trace()
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False


    def generate_context(self, context_length, depth_percents):
        context = self.read_context_files()
        context = self.encode_and_trim(context, context_length)
        context, insert_meta_data = self.insert_needle(context, depth_percents, context_length) 
        return context, insert_meta_data


    def encode_text_to_tokens(self, text):
        if self.model_provider == "Anthropic": # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            return self.enc.encode(text)
    

    def insert_needle(self, context, depth_percents, context_length): 
        # depth_percents is a tuple containing two depth percentages
        tokens_needle = [self.needle_tok1, self.needle_tok2]
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        total_needle_length = len(tokens_needle[0]) + len(tokens_needle[1])
        if len(tokens_context) + total_needle_length > context_length:
            tokens_context = tokens_context[:context_length - total_needle_length]

        # We want to make sure that we place our needles at sentence breaks so we first see what token a '.' is
        if self.model_provider in ["LLaMA", "LongLLaMA", "llama3-8b-80k"]: 
            period_tokens = [29889, 869]
        elif self.model_provider == "Mistral": 
            period_tokens = [842, 28723]
        elif self.model_provider == "GLM": 
            period_tokens = [918, 30930]
        else: period_tokens = self.encode_text_to_tokens('.')

        insertion_points = []
        for depth_percent in depth_percents:
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            insertion_points.append(insertion_point)
            print("Insertion at %d" % insertion_point)

        # Sort insertion points to insert the needles in the correct order
        insertion_points.sort()

        # Insert the needles into the context
        new_context_tokens = []
        last_insertion_point = 0
        for i, insertion_point in enumerate(insertion_points):
            new_context_tokens.extend(tokens_context[last_insertion_point:insertion_point])
            new_context_tokens.extend(tokens_needle[i])
            last_insertion_point = insertion_point

        new_context_tokens.extend(tokens_context[last_insertion_point:])
        # Convert back to a string and return it
        new_context = self.decode_tokens(new_context_tokens)
        return new_context, {
            "shortcut_key_pos_bt": 0,
            "shortcut_key_pos_ed": 0,
            "insert_points": insertion_points,
            "insert_points_lengths": [insertion_point + len(tokens_needle) for insertion_point in insertion_points]
        }


    def get_context_length_in_tokens(self, context):    
        if self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            encoded = self.enc.encode(context)
            return len(self.enc.encode(context).ids)
        else:
            return len(self.enc.encode(context))


    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while len(context.split()) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        if self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(context).ids
        else:
            return self.enc.encode(context)
        
    def decode_tokens(self, tokens, context_length=None):
        return self.enc.decode(tokens[:context_length])

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle}")
        print ("\n\n")

    def start_test(self, args):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test(args)
        auto_save_data(self.instance_counter, f"instance_score/{self.model_version}_case{self.needle_id}.pkl")
        auto_save_data(self.head_counter, f"head_score/{self.model_version}_case{self.needle_id}.pkl")
        auto_save_data(self.reference_counter, f"reference_score/{self.model_version}_case{self.needle_id}.pkl")


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', metavar='N', type=int, help='a number')
    parser.add_argument('-e', '--e_len', metavar='N', type=int, help='a number')
    parser.add_argument('-ni', '--needle_id', metavar='N', type=int, help='a number')
    parser.add_argument('--needle_file', type=str, default=None, help='/vepfs/wcf/G/zecheng/Retrieval_Head/haystack_for_detect/needle_multi_hop.jsonl')
    parser.add_argument('--model_path', type=str, default=None, help='path to model')
    parser.add_argument('--model_name_suffix', type=str, default=None, help='name of model')
    parser.add_argument('--model_provider', type=str, default="LLaMA", help='which model to use')
    parser.add_argument('--peft_model_path', type=str, default="LLaMA", help='which model to use')
    args = parser.parse_args()
   
    model_name = args.model_path

    ht = LLMNeedleHaystackTester(
        model_name=model_name, 
        model_name_suffix=args.model_name_suffix,
        model_provider=args.model_provider,
        save_contexts=False,
        save_results=True,
        needle_id=args.needle_id,
        needle_file=args.needle_file,
        document_depth_percent_intervals=6,
        peft_model_path=args.peft_model_path,
        
    )

    ht.start_test(args)
