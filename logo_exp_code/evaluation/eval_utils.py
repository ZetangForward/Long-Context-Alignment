import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
print(sys.path)
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import PeftConfig, PeftModel, PeftModelForCausalLM
import torch
from rouge_score import rouge_scorer
from utils.logn_llms.llama import LlamaForCausalLM
from utils.logn_llms.mistral import MistralForCausalLM
from modelzipper.tutils import *

"""
LONG BENCH DATA SETTING
"""
longbench_dataset_prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper_e": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "hotpotqa_e": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa_e": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "gov_report_e": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum_e": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news_e": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa_e": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum_e": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "lcc_e": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p_e": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}

longbench_pred_length = {
    "narrativeqa": 128,
    "qasper_e": 128,
    "multifieldqa_en": 64,
    "hotpotqa_e": 32,
    "2wikimqa_e": 32,
    "musique": 32,
    "qmsum_e": 512,
    "gov_report_e": 512,
    "multi_news_e": 512,
    "trec": 64,
    "triviaqa_e": 32,
    "samsum_e": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc_e": 64,
    "repobench-p_e": 64
}

DATASET2CATEGORY = {
    "narrativeqa": "EN Single-Doc QA",
    "qasper_e": "EN Single-Doc QA",
    "multifieldqa_en": "EN Single-Doc QA",
    "multifieldqa_zh": "CN Single-Doc QA",
    "hotpotqa_e": "EN Multi-Doc QA",
    "2wikimqa_e": "EN Multi-Doc QA",
    "musique": "EN Multi-Doc QA",
    "dureader": "CN Multi-Doc QA",
    "gov_report_e": "EN Summarization",
    "qmsum_e": "EN Summarization",
    "multi_news_e": "EN Summarization",
    "vcsum": "CN Summarization",
    "trec": "EN Few-Shot Learning",
    "triviaqa_e": "EN Few-Shot Learning",
    "samsum_e": "EN Few-Shot Learning",
    "lsht": "CN Few-Shot Learning",
    "passage_retrieval_en": "EN Synthetic Task",
    "passage_count": "EN Synthetic Task",
    "passage_retrieval_zh": "CN Synthetic Task",
    "lcc_e": "Code Completion",
    "repobench-p_e": "Code Completion",
}

all_datasets = [
    "qasper_e", "multifieldqa_en", "hotpotqa_e", "2wikimqa_e", "gov_report_e", 
    "multi_news_e", "musique", "trec", "triviaqa_e",  "samsum_e", "passage_count", 
    "passage_retrieval_en", "lcc_e", "repobench-p_e", "narrativeqa", "qmsum_e",
]


def load_model(model_type, model_path, adapter_path = None, rope_theta=None, rope_factor=None, rope_type=None, max_position_embeddings: int = None, max_testing_length: int = None, max_training_length: int = 8192, use_logn: bool=False, device: str = None):
    
    config = AutoConfig.from_pretrained(model_path)
    
    if rope_type is not None:
        config.rope_scaling = {"type": rope_type, "factor": rope_factor}
    if rope_theta is not None and rope_theta != -1:
        print_c(f'Using rope_theta: {rope_theta}')
        config.rope_theta = rope_theta
    if max_position_embeddings is not None:
        config.max_position_embeddings = max_position_embeddings

    if use_logn:
        log_c("Using logn scaling...")
        config.use_logn_scaling = True
        config.max_testing_length = max_testing_length
        config.max_training_length = max_training_length

        if model_type.lower() in ('llama-3', 'llama-2', 'llama2', 'llama3'):
            model = LlamaForCausalLM.from_pretrained(
                model_path, config=config, attn_implementation="flash_attention_2", 
                torch_dtype=torch.bfloat16
            ).to(device)
        elif model_type.lower() == 'mistral':
            model = MistralForCausalLM.from_pretrained(
                model_path, config=config, attn_implementation="flash_attention_2", 
                torch_dtype=torch.bfloat16
            ).to(device)
    
    else:
        log_c("Using vanilla transformers implementation...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config, attn_implementation="flash_attention_2", 
            torch_dtype=torch.bfloat16, 
        ).to(device)
    
    if adapter_path is not None:    
        log_c("Using adapter Peft...")
        model = PeftModelForCausalLM.from_pretrained(model, adapter_path)
    
    return model

def load_old_model(model_path, adapter_path = None, rope_theta=None, rope_factor=None, rope_type=None, max_position_embeddings: int = None):
    
    config = AutoConfig.from_pretrained(model_path)
    if rope_type is not None:
        config.rope_scaling = {"type": rope_type, "factor": rope_factor}
    if rope_theta is not None:
        config.rope_theta = rope_theta
    if max_position_embeddings is not None:
        config.max_position_embeddings = max_position_embeddings

    if adapter_path is not None:    
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config,  
            torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModelForCausalLM.from_pretrained(model, adapter_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=config,
            torch_dtype=torch.bfloat16, device_map="auto"
        )
    return model



class CustomScorer:
    def __init__(self, real_answer, forbidden_strings):
        self.real_answer = real_answer
        self.forbidden_strings = forbidden_strings
        self.scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        self.penalty_factor = 100 / (len(self.forbidden_strings) + 1)

    def score(self, response):
        # Calculate initial ROUGE-1 recall score
        initial_scores = self.scorer.score(self.real_answer, response)
        initial_recall_score = initial_scores['rouge1'].recall * 100
        
        # Calculate forbidden strings penalty
        forbidden_penalty_nums = self.count_forbidden_entity_nums(response)
        penalty_value = forbidden_penalty_nums * self.penalty_factor

        # Calculate length penalty
        # length_penalty = self.calculate_length_penalty(response)
        
        # Apply penalties to the ROUGE-1 recall score
        penalized_recall_score = initial_recall_score - penalty_value
        
        return initial_recall_score, penalized_recall_score

    def calculate_length_penalty(self, response):
        # Define a base penalty factor for length
        length_factor = 0.01
        response_length = len(response.split())
        length_penalty = length_factor * response_length
        return min(1, length_penalty)  # Ensure the penalty is between 0 and 1

    def count_forbidden_entity_nums(self, response):
        penalty_count = 0
        
        for forbidden in self.forbidden_strings:
            if forbidden.lower() in response.lower():
                penalty_count += 1
        
        return penalty_count