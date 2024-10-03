
#import tiktoken
import os 
import glob
import json
from modelzipper.tutils import *
from transformers import AutoTokenizer, AutoConfig
import sys
sys.path.append("./faiss_attn/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from source.modeling_llama import LlamaForCausalLM
from source.modeling_qwen2 import Qwen2ForCausalLM
from source.modeling_mixtral import MixtralForCausalLM
from source.modeling_mistral import MistralForCausalLM
from source.modeling_self_extend_llama import SelfExtendLlamaForCausalLM
import numpy as np
import argparse
from rouge_score import rouge_scorer
from datetime import datetime, timezone
from collections import defaultdict
import time
import torch
from peft import LoraConfig, get_peft_model, PeftModelForCausalLM


def reset_rope(model, model_max_train_len, scaling_factor):
    for l in model.model.layers:
        l.self_attn.rotary_emb.scaling_factor = scaling_factor
        l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=model_max_train_len, device=l.self_attn.rotary_emb.inv_freq.device, dtype=torch.float32)
    
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(
            self, haystack_dir="./haystack_for_detect", results_version = 1, context_lengths_min = 1000, context_lengths_max = 50000, 
            context_lengths_num_intervals = 20, context_lengths = None, document_depth_percent_min = 0, document_depth_percent_max = 100, document_depth_percent_intervals = 10, document_depth_percents = None, document_depth_percent_interval_type = "linear", model_name='', model_name_suffix=None, num_concurrent_requests = 1, save_results = True, save_contexts = True, final_context_length_buffer = 200, peft_model_path=None, seconds_to_sleep_between_completions = None, needle_file = None, insert_short_key_id = 0, needle_id = 0, shortcut_strategy = "random", topk=1, **kwargs
        ):
        needles_and_stacks = auto_read_data(f"{haystack_dir}/{needle_file}")
        self.shortcut_keys = auto_read_data(f"{haystack_dir}/shortcut_key.jsonl")
        self.needle_list = [l["needle"] for l in needles_and_stacks]
        self.haystack_dir_list = [f"{haystack_dir}/part{i}" for i in range(1, 4)]
        self.retrieval_question_list = [l["question"] for l in needles_and_stacks]
        self.real_ansers_list = [l["real_needle"] for l in needles_and_stacks]
        self.shortcut_key = (self.shortcut_keys[insert_short_key_id]["prefix"], self.shortcut_keys[insert_short_key_id]["suffix"])
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.shortcut_strategy = shortcut_strategy
        self.testing_results = []
        self.topk = topk
        self.head_counter = defaultdict(list)
        self.reference_counter = defaultdict(list)
        self.instance_counter = defaultdict(list)
        self.peft_model_path = peft_model_path
        if("/" in model_name):
            self.model_version = model_name.split("/")[-1]
        else: self.model_version = model_name
        
        if(model_name_suffix is not None): 
            self.model_version += "_" + model_name_suffix

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        self.model_name = model_name
        self.enc = AutoTokenizer.from_pretrained(model_name)

        print("loading from %s" % model_name)
        self.load_model()
            
        if "CUDA_VISIBLE_DEVICES" in os.environ: 
            self.multi_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"])>1
        else: 
            self.multi_gpus = True
        
        self.needle_id = needle_id
        self.haystack_dir = self.haystack_dir_list[0]
        self.needle = self.needle_list[needle_id]
        self.needle_tok = self.enc(self.needle).input_ids[1:]
        self.real_answer = self.real_ansers_list[needle_id]
        self.real_answer_tok = self.enc(self.real_answer).input_ids[1:]
        self.retrieval_question = self.retrieval_question_list[needle_id]
        self.shortcut_prefix_tok = self.enc(self.shortcut_key[0]).input_ids[1:] if len(self.shortcut_key) > 0 else None
        self.shortcut_suffix_tok = self.enc(self.shortcut_key[0]).input_ids[1:] if len(self.shortcut_key) > 0 else None
        self.model_to_test_description = model_name
        model_name = model_name.split('/')[-1]


    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    

    def load_model(self):
        config = AutoConfig.from_pretrained(model_name)
        print("loading from %s successfully ..." % self.model_version)

        if "Mistral-7B" in self.model_version:
            self.model_to_test = MistralForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16,device_map='auto',use_flash_attention_2="flash_attention_2",trust_remote_code=True).eval()

        elif 'Llama-3-8B-Instruct-80K' in self.model_version:
            config.rope_theta = 200e6
            self.model_to_test = LlamaForCausalLM.from_pretrained(model_name, config=config, use_flash_attention_2="flash_attention_2", torch_dtype=torch.bfloat16, device_map='auto').eval()
        
        if self.peft_model_path is not None:
            self.model_to_test = PeftModelForCausalLM.from_pretrained(self.model_to_test, self.peft_model_path)
        
        self.layer_num, self.head_num = config.num_hidden_layers, config.num_attention_heads
        print(f"layer number: {self.layer_num}, head number {self.head_num}")

    def run_test(self, args):
        for context_length in self.context_lengths:
            if context_length < args.s_len or context_length > args.e_len: 
                continue
            for depth_percent in self.document_depth_percents:
                self.evaluate_and_log(context_length, depth_percent)


    def retrieval_calculate(self, attention_maxtrix, attribution_scores, inp, step_token, citation_scores, step_i, topk=1):
        for layer_idx in range(self.layer_num):
            for head_idx in range(self.head_num):
                values, idx = attention_maxtrix[layer_idx][0][head_idx][-1].topk(topk)  # find the topk attention value and index   
                
                # citation scores 区别于下面，这里是一个更粗粒度的范围
                for v, i in zip(values, idx):  
                    if self.reference_start <= i < self.reference_end:  
                        citation_scores[layer_idx][head_idx][step_i].append((i.item(), v.item())) 
                    else:
                        citation_scores[layer_idx][head_idx][step_i].append((i.item(), 0))       
                
                # attribution scores 表示一个细粒度的召回，hit了实体的位置
                for v, i in zip(values, idx):  # find the top k attention id for the needle hit
                    if self.needle_start <= i < self.needle_end and inp.item()==self.prompt_ids[i].item():
                        attribution_scores[layer_idx][head_idx][0] += 1/(self.needle_end - self.needle_start)  # 解释：hit了多少needle，比如neddle是两个index：[232, 321, 876]，模型hit了其中两个（假设是232，321），那么就是 + 1/3 + 1/3，对于频率，只要这里的[0]位置大于0，就算一次成功，不要求hit到1
                        attribution_scores[layer_idx][head_idx][1] += step_token
                        break  # 这里可能设置topk=5，只要有一个hit到就可以了，这个可以做pre-2（搜索空间的实验）
    

    def citation_head_accumulate(self, reference_score, recall_score):
        for layer_idx in range(self.layer_num):
            for head_idx in range(self.head_num): # per head per layer save the reference score
                self.reference_counter[f"{layer_idx}-{head_idx}"].append((  # save all depth and context
                    reference_score[layer_idx][head_idx], # reference attention id and attention score
                    recall_score)) # recall score
    

    def attribute_head_accumulate(self, retrieval_score, recall_score):
        for layer_idx in range(self.layer_num):
            for head_idx in range(self.head_num):
                self.head_counter[f"{layer_idx}-{head_idx}"].append((
                    retrieval_score[layer_idx][head_idx][0], # retrieval score, ignore the pred token
                    recall_score)) # recall score


    def instance_score_accumulate(self, hit_candidate_scores, entropy_scores, context_length, depth_percent):
        self.instance_counter[f'{context_length}-{depth_percent}'].append((hit_candidate_scores, entropy_scores))


    def decode(self, q_outputs, inp, decode_len, block_list=None):
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
            topk_values, topk_indices = torch.topk(outputs.logits[0, -1], 1, dim=-1)
            gen_candidates.append((topk_indices, topk_values))
            inp = outputs.logits[0, -1].argmax()
            step_token = self.enc.convert_ids_to_tokens(inp.item())
            output.append(inp.item())
            self.retrieval_calculate(
                attention_maxtrix=outputs.attentions, 
                attribution_scores=attribution_scores, 
                inp=inp, step_token=step_token, 
                citation_scores=citation_scores, 
                step_i=step_i,
                topk=self.topk,
            )

        return {
            "model_generation": output, 
            "attribution_scores": attribution_scores, 
            "hit_candidate_scores": gen_candidates,
            "citation_scores": citation_scores, 
            "needle_entropy": needle_entropy,  # 暂时不考虑信息熵的事情
        }


    def find_needle_idx(self, needle):
        needle_ids = self.enc(needle, add_special_tokens=False)["input_ids"]
        span_len = len(needle_ids)
        for i in range(len(self.prompt_ids)):            
            token_span = self.prompt_ids[i : i + span_len]
            span_ids = set(token_span.tolist())
            overlap = float(len(span_ids.intersection(set(needle_ids)))) / len(set(needle_ids))
            if(overlap > 0.9):
                return i, i + span_len
        return -1, -1
    

    def find_citation_idx(self, reference):
        needle_ids = self.enc(reference, add_special_tokens=False)["input_ids"]
        span_len = len(needle_ids)
        for i in range(len(self.prompt_ids)):            
            token_span = self.prompt_ids[i : i + span_len]
            span_ids = set(token_span.tolist())
            overlap = float(len(span_ids.intersection(set(needle_ids)))) / len(set(needle_ids))
            if(overlap > 0.9):
                return i, i + span_len
        return -1, -1
    
    
    def evaluate_and_log(self, context_length, depth_percent):
        context, _ = self.generate_context(context_length, depth_percent)
        prompt = [{"role": "user", "content": f"{context}\n Question: {self.retrieval_question}\nAnswer:"},]
        input_ids = self.enc.apply_chat_template(
            conversation=prompt, tokenize=True, add_generation_prompt=True, return_tensors='pt'
        )
     
        # Prepare your message to send to the model you're going to evaluate
        test_start_time = time.time()
        self.prompt_ids = input_ids[0, :]
        if not self.multi_gpus:
            input_ids = input_ids.to(self.model_to_test.device)
        self.needle_start, self.needle_end = self.find_needle_idx(self.real_answer)
        self.reference_start, self.reference_end = self.find_citation_idx(self.needle)
        
        with torch.no_grad():
            q_outputs = self.model_to_test(input_ids=input_ids[:,:-1], use_cache=True, return_dict=True)
            decode_res = self.decode(q_outputs, input_ids[:,-1], len(self.real_answer_tok) + 32) 
            response = self.enc.decode(decode_res["model_generation"], skip_special_tokens=True).strip()

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        recall_score = scorer.score(self.real_answer, response)['rouge1'].recall
        
        self.citation_head_accumulate(decode_res['citation_scores'], recall_score)
        self.attribute_head_accumulate(decode_res['attribution_scores'], recall_score)
        self.instance_score_accumulate(decode_res['hit_candidate_scores'], decode_res['needle_entropy'], context_length, depth_percent)
     
        # head_score = [(i[0], np.mean(i[1])) for i in self.head_counter.items()]
        # head_score = sorted(head_score, key=lambda x: x[1], reverse=True)
        # print([[i[0]] for i in head_score][:15])

        results = {
            'model' : self.model_to_test_description,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'question': self.retrieval_question,
            'model_response' : response,
            'score' : recall_score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }

        self.testing_results.append(results)

        print (f"-- Test Summary -- ")
        print (f"Duration: {test_elapsed_time:.1f} seconds")
        print (f"Context: {context_length} tokens")
        print (f"Depth: {depth_percent}%")
        print (f"Score: {recall_score}")
        print (f"Response: {response}\n")

        context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'

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


    def generate_context(self, context_length, depth_percent):
        context = self.read_context_files()
        context = self.encode_and_trim(context, context_length)
        context, insert_meta_data = self.insert_needle(context, depth_percent, context_length) 
        return context, insert_meta_data


    def insert_needle_shortcut(self, context, depth_percent, context_length):  # insert both shortcut and needle
        tokens_needle = self.needle_tok
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle + shortcut keys are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) + len(self.shortcut_key_tok) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle) - len(self.shortcut_key_tok)]

        # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
        period_tokens = self.encode_text_to_tokens('.')

        if depth_percent == 100:
            if self.shortcut_strategy == "random":
                shortcut_key_position = random.randint(self.final_context_length_buffer, len(tokens_context) - 1)
                tokens_new_context = tokens_context[:shortcut_key_position]
                # insert shortcut key in random position, before a whole sequence
                while tokens_new_context and tokens_new_context[-1] not in period_tokens:  
                    shortcut_key_position -= 1
                    tokens_new_context = tokens_context[:shortcut_key_position]
                tokens_new_context += self.shortcut_key_tok + tokens_context[:shortcut_key_position] + tokens_needle
            elif self.shortcut_strategy == "before":
                tokens_new_context = tokens_context + self.shortcut_key_tok + tokens_needle
                shortcut_key_position = len(tokens_new_context) - len(tokens_needle)
            insertion_point = len(tokens_new_context) - len(tokens_needle)
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            print("insertion at %d" % insertion_point)
        
            if self.shortcut_strategy == "random":
                tokens_new_context += tokens_needle + tokens_context[insertion_point:]
                if self.shortcut_position == 0: 
                    short_pos_st, short_pos_ed = self.final_context_length_buffer, insertion_point
                elif self.shortcut_position == 1: 
                    short_pos_st, short_pos_ed = insertion_point + len(tokens_needle), len(tokens_new_context) - 1
                
                if short_pos_st > short_pos_ed:  # can only insert shortcut key after the needle
                    self.shortcut_position = 1
                    short_pos_st, short_pos_ed = insertion_point + len(tokens_needle), len(tokens_new_context) - 1

                # Insert Shortcut squence
                shortcut_key_position = random.randint(short_pos_st, short_pos_ed)
                prefix, suffix = tokens_new_context[:shortcut_key_position], tokens_new_context[shortcut_key_position:]
                if self.shortcut_position == 0: # insert in the left, shift to left position 
                    while suffix and suffix[0] not in period_tokens:  # insert shortcut key before a whole sequence
                        shortcut_key_position -= 1 
                        prefix, suffix = tokens_new_context[:shortcut_key_position], tokens_new_context[shortcut_key_position:]
                else: # insert in the right, shift to right position  
                    while suffix and prefix[-1] not in period_tokens:  
                        shortcut_key_position += 1 
                        prefix, suffix = tokens_new_context[:shortcut_key_position], tokens_new_context[shortcut_key_position:]
                tokens_new_context = prefix + self.shortcut_key_tok + suffix
            elif self.shortcut_strategy == "before":
                tokens_new_context += self.shortcut_key_tok + tokens_needle + tokens_context[insertion_point:]
                shortcut_key_position = insertion_point
            elif self.shortcut_strategy == "after":
                tokens_new_context += self.shortcut_key_tok + tokens_needle + tokens_context[insertion_point:]
                shortcut_key_position = insertion_point
            else:
                raise NotImplementedError

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context, {"shortcut_key_pos_bt": shortcut_key_position, "shortcut_key_pos_ed": shortcut_key_position + len(self.shortcut_key_tok), "insert_point_bt": insertion_point, "insert_point_ed": insertion_point+len(tokens_needle)}

    
    def encode_text_to_tokens(self, text):
        return self.enc.encode(text)
     
    
    def insert_needle(self, context, depth_percent, context_length):  # just insert needle
        tokens_needle = self.needle_tok
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle + shortcut keys are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
        period_tokens = self.encode_text_to_tokens('.')

        if depth_percent == 100:
            tokens_new_context = tokens_context + tokens_needle
            insertion_point = len(tokens_new_context) - len(tokens_needle)
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            print("insertion at %d" % insertion_point)
        
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context, {"shortcut_key_pos_bt": 0, "shortcut_key_pos_ed": 0, "insert_point_bt": insertion_point, "insert_point_ed": insertion_point+len(tokens_needle)}

    def get_context_length_in_tokens(self, context):       
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
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self, args):
        self.print_start_test_summary()
        self.run_test(args)
        auto_save_data(self.reference_counter, f"reference_score/{self.model_version}_case{self.needle_id}.pkl")


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', default=0, metavar='N', type=int, help='a number')
    parser.add_argument('-e', '--e_len', default=50000, metavar='N', type=int, help='a number')
    parser.add_argument('-ni', '--needle_id', default=0, metavar='N', type=int, help='a number')
    parser.add_argument('-di', '--document_depth_percent_intervals', default=0, metavar='N', type=int, help='a number')
    parser.add_argument('-ci', '--context_lengths_num_intervals', default=0, metavar='N', type=int, help='a number')
    parser.add_argument('--topk', default=1, metavar='N', type=int, help='topk value')
    parser.add_argument('--needle_file', type=str, default='needles.jsonl', help='path to needle file')
    parser.add_argument('--model_path', type=str, default='/data/zecheng/hf_models/llama-2-7b-80k', help='path to model')
    parser.add_argument('--model_name_suffix', type=str, default='NIAH', help='name of model')
    parser.add_argument('--peft_model_path', type=str, default=None, help='peft model path')
    args = parser.parse_args()
   
    model_name = args.model_path
    
    ht = LLMNeedleHaystackTester(
        model_name=model_name, 
        model_name_suffix=args.model_name_suffix,
        save_contexts=False,
        save_results=False,
        needle_id=args.needle_id,
        needle_file=args.needle_file,
        topk=args.topk,
        context_lengths_min=args.s_len,
        context_lengths_max=args.e_len,
        peft_model_path=args.peft_model_path,
        document_depth_percent_intervals=args.document_depth_percent_intervals,
        context_lengths_num_intervals=args.context_lengths_num_intervals
    )

    ht.start_test(args)
