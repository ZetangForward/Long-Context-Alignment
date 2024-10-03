import os
import math
import sys
import torch
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import glob
import pytz
from datetime import datetime
from rouge import Rouge
from typing import List, Optional
from tqdm import trange, tqdm
from transformers import HfArgumentParser, AutoTokenizer
from transformers.utils import logging
from dataclasses import dataclass, field, asdict
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from eval_utils import all_datasets, longbench_dataset_prompt, longbench_pred_length, load_model, DATASET2CATEGORY
from chat import apply_chat_template
from modelzipper.tutils import *


logger = logging.get_logger(__name__)


class FileLogger:
    def __init__(self, log_file) -> None:
        self.log_file = log_file
    
    def log(self, metrics, **kwargs):
        with open(self.log_file, "a+") as f:
            # get current time
            tz = pytz.timezone('Asia/Shanghai')
            time = f"{'Time': <10}: {json.dumps(datetime.now(tz).strftime('%Y-%m-%d, %H:%M:%S'), ensure_ascii=False)}\n"
            print(time)
            command = f"{'Command': <10}: {json.dumps(' '.join(sys.argv), ensure_ascii=False)}\n"
            print(command)
            metrics = f"{'Metrics': <10}: {json.dumps(metrics, ensure_ascii=False)}\n"
            msg = time + command

            for key, value in kwargs.items():
                x = f"{key: <10}: {json.dumps(value, ensure_ascii=False)}\n"
                print(x)
                msg += x
            msg += metrics
            print(metrics)
            f.write(str(msg) + "\n")

def get_rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores

@dataclass
class Args:
    haystack_path: str = field(
        default="long-llm:needle/PaulGrahamEssays",
        metadata={'help': 'The context for evaluation.'}
    )
    
    result_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'The directory relative to output_dir for saving results.'}
    )

    min_length: int = field(
        default=8192,
        metadata={'help': 'Minimum context length in evaluation.'}
    )
    max_length: int = field(
        default=131072,
        metadata={'help': 'Maximum context length in evaluation.'}
    )
    num_length_interval: int = field(
        default=10,
        metadata={'help': 'Number of invervals between min_length and max_length.'}
    )
    test_length: List[int] = field(
        default=None,
        metadata={'help': 'Specified evaluation lengths.'}
    )

    min_depth: float = field(
        default=0,
        metadata={'help': 'Minimum pass key depth in the context.'}
    )
    max_depth: float = field(
        default=100,
        metadata={'help': 'Maximum pass key depth in the context.'}
    )
    num_depth_interval: int = field(
        default=10,
        metadata={'help': 'Number of invervals between min_depth and max_depth.'}
    )
    test_depth: List[int] = field(
        default=None,
        metadata={'help': 'Specified evaluation depths.'}
    )

    needle: str = field(
        default="\n\nThe best thing to do in San Francisco is sitting in Dolores Park and eating a hamburg on a sunny day.\n\n",
        metadata={'help': 'The needle content'}
    )
    prompt: str = field(
        default='\n\nWhat is the best thing to do in San Francisco?\nAnswer:',
        metadata={'help': 'The needle content'}
    )
    answer: str = field(
        default='sitting in Dolores Park and eating a hamburg on a sunny day',
    )
    gpt_eval: bool = field(
        default=False,
        metadata={'help': 'Use GPT4 to evaluate accuracy.'}
    )
    proxy: Optional[str] = field(
        default=None,
        metadata={'help': 'Proxy when using gpt evaluation.'}
    )

    load_result: bool = field(
        default=False,
        metadata={'help': 'Load previous results?'}
    )

    rouge: str = field(
        default='rouge-l',
    )
    eva_indic: str = field(
        default='p',
    )
    
    peft_path: Optional[str] = field(
        default=None,
        metadata={'help': 'adapter path'}
    )
    
    rope_type: Optional[str] = field(
        default=None,
        metadata={'help': 'rope_type'}
    )
    
    rope_factor: Optional[float] = field(
        default=None,
        metadata={'help': 'rope_factor'}
    )

    rope_theta: Optional[float] = field(
        default=None,
        metadata={'help': 'rope_theta'}
    )
    
    max_position_embeddings: int = field(
        default=None,
        metadata={'help': 'Number of invervals between min_depth and max_depth.'}
    )

    do_sample: bool = False
    
    model_path: str = None
    
    chat_template: str = None


class OpenAIEvaluator:
    DEFAULT_MODEL_KWARGS: dict = dict(temperature=0)
    CRITERIA = {"accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numberical score"""}

    def __init__(self,
                 model_name: str = "gpt-3.5-turbo-0125",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                 true_answer: str = None,
                 question_asked: str = None,
                 proxy: str = None):
        """
        :param model_name: The name of the model.
        :param model_kwargs: Model configuration. Default is {temperature: 0}
        :param true_answer: The true answer to the question asked.
        :param question_asked: The question asked to the model.
        """
        # from langchain_openai import ChatOpenAI
        from langchain_community.chat_models import ChatOpenAI

        if (not true_answer) or (not question_asked):
            raise ValueError("true_answer and question_asked must be supplied with init.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.true_answer = true_answer
        self.question_asked = question_asked
        self.proxy = proxy

        api_key = os.getenv('OPENAI_API_KEY')
        if (not api_key):
            raise ValueError("OPENAI_API_KEY must be in env for using openai evaluator.")

        self.api_key = api_key
        
        self.evaluator = ChatOpenAI(model=self.model_name,
                                    openai_api_key=self.api_key,
                                    openai_proxy=self.proxy,
                                    **self.model_kwargs)

    def evaluate_response(self, response: str) -> int:
        from langchain.evaluation import load_evaluator

        evaluator = load_evaluator(
            "labeled_score_string",
            criteria=self.CRITERIA,
            llm=self.evaluator,
        )

        eval_result = evaluator.evaluate_strings(
            # The models response
            prediction=response,

            # The actual answer
            reference=self.true_answer,

            # The question asked
            input=self.question_asked,
        )

        return int(eval_result['score'])


def generate_sample(
    tokenizer, 
    chat_template, 
    context, 
    context_length, 
    needle_depth, 
    needle="\n\nThe best thing to do in San Francisco is sitting in Dolores Park and eating a hamburg on a sunny day.\n\n", 
    prompt='\n\nWhat is the best thing to do in San Francisco?\nAnswer:'):
    
    num_words = len(context.split())
    if context_length > num_words:
        context = context * math.ceil(context_length / num_words)

    description = "There is an important infomation hidden in the following context. Find the information and memorize it. I will quiz you about the important information there.\n"

    description_input_ids = tokenizer.encode(description, add_special_tokens=False)
    needle_input_ids = tokenizer.encode(needle, add_special_tokens=False)
    prompt_input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    description_length = len(description_input_ids)
    needle_length = len(needle_input_ids)
    prompt_length = len(prompt_input_ids)

    # must leave room for information and prompt
    minimum_pos = description_length
    maximum_pos = context_length - prompt_length - needle_length - 1
    if minimum_pos > context_length or maximum_pos < 0:
        raise ValueError(f"The length {context_length} is too small. Please increase interval!")

    needle_pos = minimum_pos + round((maximum_pos - minimum_pos) * needle_depth / 100)
    
    context_input_ids = tokenizer.encode(context, max_length=context_length - description_length - needle_length - prompt_length, truncation=True, add_special_tokens=False)

    input_ids = sum([description_input_ids, context_input_ids[:needle_pos], needle_input_ids, context_input_ids[needle_pos:], prompt_input_ids], [])
    inputs = tokenizer.decode(input_ids)

    inputs = apply_chat_template(chat_template, messages=[{'role': 'user', 'content': inputs}], tokenizer=tokenizer, add_generation_prompt=True).raw

    return inputs, prompt, needle


def main():
    parser = HfArgumentParser([Args])
    args = parser.parse_args_into_dataclasses()[0]

    if args.load_result:
        with open(os.path.join(args.result_dir, "results.json"), "r", encoding='utf-8') as f:
            results = json.load(f)

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        if args.peft_path is not None and args.peft_path == "-1":
            args.peft_path = None
            
        model = load_model(
            model_type='no', model_path=args.model_path, 
            max_position_embeddings = args.max_position_embeddings,
            adapter_path=args.peft_path, rope_type=args.rope_type, 
            rope_theta = args.rope_theta, rope_factor=args.rope_factor, device='cuda'
        )
        if args.test_length is None:
            test_lengths = np.linspace(args.min_length, args.max_length, args.num_length_interval, endpoint=True).astype(int).tolist()
        else:
            test_lengths = args.test_length

        if args.test_depth is None:
            test_depths = np.linspace(args.min_depth, args.max_depth, args.num_depth_interval, endpoint=True).astype(int).tolist()
        else:
            test_depths = args.test_depth

        logger.info('testing length is:')
        logger.info(test_lengths)
        logger.info('testing depth is:')
        logger.info(test_depths)
        
        if os.path.isfile(args.haystack_path):
            with open(args.haystack_path) as f:
                context = f.read().strip()
        elif os.path.isdir(args.haystack_path):
            context = ""
            num_tokens = 0
            for file in glob.glob(f"{args.haystack_path}/*.txt"):
                with open(file, 'r') as f:
                    this_file_context = f.read()
                    num_tokens += len(tokenizer.encode(this_file_context, add_special_tokens=False))
                    context += this_file_context
                    if num_tokens > max(test_lengths):
                        break
        else:
            raise ValueError(f"Cannot find haystack: {args.haystack_path}")

        all_inputs = []
        for length in tqdm(test_lengths, desc="Constructing Data"):
            for depth in test_depths:
                inputs, prompt, needle = generate_sample(
                    tokenizer=tokenizer, 
                    chat_template=args.chat_template, 
                    context=context,
                    context_length=length, 
                    needle_depth=depth,
                    needle=args.needle,
                    prompt=args.prompt
                )
                all_inputs.append({'inputs': inputs, 'prompt': prompt, 'needle': needle, 'length': length, 'depth': depth})
        
        all_inputs = all_inputs[::-1]
        results = {l: {d: [] for d in test_depths} for l in test_lengths}
        
        with torch.no_grad():
            for i in trange(len(all_inputs), desc="Evaluating"):

                if hasattr(model, "memory"):
                    model.memory.reset()

                x = all_inputs[i]
                prompt = x.pop("prompt")
                inputs = x.pop("inputs")
                inputs = tokenizer(inputs, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.,
                )
                outputs = outputs[:, inputs['input_ids'].shape[1]:].contiguous()
                
                o = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                results[x['length']][x['depth']].append({'target': x['needle'], 'prediction': o})
                print(f'Length: {x["length"]}, Depth: {x["depth"]}\nPrediction: {o}\nTarget: {x["needle"]}')
                print('-' * 20)
            
        auto_save_data(results, f'{args.result_dir}/results.json')
        
    rouge_score = {l: {d: [] for d in v.keys()} for l, v in results.items()}
    if args.gpt_eval:
        evaluator = OpenAIEvaluator(question_asked=args.prompt.strip(), true_answer=args.needle.strip(), proxy=args.proxy)
        gpt_score = {l: {d: [] for d in v.keys()} for l, v in results.items()}

    logger.info('begin to calculate metrics')
    
    for l, lv in results.items():
        for d, dv in lv.items():
            for v in dv:
                prediction = v["prediction"]
                score = get_rouge_score(prediction.lower().strip().split('\n')[0], args.answer.lower())
                score = score[args.rouge][args.eva_indic]
                rouge_score[l][d].append(score)

                if args.gpt_eval:
                    gpt_score[l][d].append(evaluator.evaluate_response(prediction))

            rouge_score[l][d] = round(sum(rouge_score[l][d]) / len(dv), 2)
            if args.gpt_eval:
                while 1:
                    try:
                        gpt_score[l][d] = round(sum(gpt_score[l][d]) / len(dv), 2)
                        break
                    except ValueError:
                        pass

    metrics = {'rouge': rouge_score}
    if args.gpt_eval:
        metrics["gpt"] = gpt_score
    file_logger = FileLogger(os.path.join(args.result_dir, "metrics.log"))
    file_logger.log(metrics, Args=asdict(args))

    for metric_key, metric_value in metrics.items():
        # Copied from https://github.com/gkamradt/LLMTest_NeedleInAHaystack/blob/main/viz/CreateVizFromLLMTesting.ipynb
        cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
        # Create the heatmap with better aesthetics
        plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
        data = pd.DataFrame(metric_value)

        if metric_key == "rouge":
            vmin = 0
            vmax = 1
        elif metric_key == "gpt":
            vmin = 1
            vmax = 10

        sns.heatmap(
            data,
            fmt="g",
            cmap=cmap,
            cbar_kws={'label': metric_key},
            vmin=vmin,
            vmax=vmax,
        )

        # More aesthetics
        plt.title('Needle In A HayStack')  # Adds a title
        plt.xlabel('Token Limit')  # X-axis label
        plt.ylabel('Depth Percent')  # Y-axis label
        plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
        plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
        plt.tight_layout()  # Fits everything neatly into the figure area

        plt.savefig(os.path.join(args.result_dir, f"{args.rouge+'_' +args.eva_indic}.pdf"), format='pdf', dpi=1200, bbox_inches='tight')


if __name__ == "__main__":
    main()
