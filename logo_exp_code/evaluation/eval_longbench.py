import os
import sys
import json
import argparse
import numpy as np
from fire import Fire
from modelzipper.tutils import *
from eval_utils import *
from metrics import (
    qa_f1_score, rouge_score, classification_score, 
    retrieval_score, count_score, code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper_e": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa_e": qa_f1_score,
    "2wikimqa_e": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report_e": rouge_score,
    "multi_news_e": rouge_score,
    "trec": classification_score,
    "triviaqa_e": qa_f1_score,
    "samsum_e": rouge_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "lcc_e": code_sim_score,
    "repobench-p_e": code_sim_score,
    "qmsum_e": rouge_score,
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = dict()
    total_score = 0.0
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if '[/INST]' in prediction:
            prediction = prediction.replace('[/INST]', '')
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    scores["total_score"] = round(100 * total_score / len(predictions), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if '[/INST]' in prediction:
            prediction = prediction.replace('[/INST]', '')
        if dataset in ["trec", "triviaqa_e", "samsum_e", "lsht", 'narrativeqa', '']:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def main(pred_path: str = None, benchmark_dataset_path: str = None):
    scores = dict()
    all_files = os.listdir(pred_path)
    print("Evaluating on:", all_files)

    ## extract all golden data classes
    data_classes = {}
    if benchmark_dataset_path is None:
        benchmark_dataset_path = "./data"
    all_benchmark_data_files = os.listdir(benchmark_dataset_path)
    for f in all_benchmark_data_files:
        data_name = f.split('.')[0]
        content = auto_read_data(os.path.join(benchmark_dataset_path, f))
        data_classes[data_name] = content[0]["all_classes"]
    
    ## read all predicted datasets
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        dataset_name = filename.split('.')[0]
        pred_dataset_length, all_classes = longbench_pred_length[dataset_name], data_classes[dataset_name]
        predictions, answers = [], []
       
        with open(os.path.join(pred_path, filename), "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred_str"])
                answers.append(data["answers"])

        if len(predictions) == 0: continue
        score = scorer(dataset_name, predictions, answers, all_classes)
        scores[dataset_name] = score
    
    # 假设您的JSON数据存储在变量json_data中
    json_data = scores

    # 将JSON数据转换为DataFrame
    df = pd.DataFrame({k: v for k, v in json_data.items()}, index=[0])

    # 定义新的列顺序
    columns = [
        ['Single-Document QA', 'Single-Document QA', 'Single-Document QA', 'Single-Document QA',
        'Multi-Document QA', 'Multi-Document QA', 'Multi-Document QA', 'Multi-Document QA',
        'Summarization', 'Summarization', 'Summarization', 'Summarization',
        'Few-shot Learning', 'Few-shot Learning', 'Few-shot Learning', 'Few-shot Learning',
        'Synthetic Tasks', 'Synthetic Tasks', 'Synthetic Tasks',
        'Code Completion', 'Code Completion', 'Code Completion',
        'ALL Avg'],
        ['qasper_e', 'multifieldqa_en', 'narrativeqa', 'Avg.',
        'hotpotqa_e', '2wikimqa_e', 'musique', 'Avg.',
        'gov_report_e', 'qmsum_e', 'multi_news_e', 'Avg.',
        'trec', 'triviaqa_e', 'samsum_e', 'Avg.',
        'passage_count', 'passage_retrieval_en', 'Avg.',
        'lcc_e', 'repobench-p_e', 'Avg.',
        '']
    ]

    # 重新组织数据
    new_df = pd.DataFrame(columns=columns)
    for col_0, col in zip(columns[0], columns[1]):
        if col in df.columns:
            new_df[(col_0, col)] = df[col]
        elif col == 'Avg.':
            new_df[(col_0, col)] = np.nan
        else:
            new_df[(col_0, col)] = np.nan

    # 计算平均值
    for category in ['Single-Document QA', 'Multi-Document QA', 'Summarization', 'Few-shot Learning', 'Synthetic Tasks', 'Code Completion']:
        mask = new_df.columns.get_level_values(0) == category
        new_df.loc[:, (category, 'Avg.')] = new_df.loc[:, mask].mean(axis=1)

    # 计算总平均值
    new_df[('ALL Avg', '')] = new_df.loc[:, new_df.columns.get_level_values(1) == 'Avg.'].mean(axis=1)
    out_path = os.path.join(pred_path, "result.csv")
    # 保存为 CSV 文件
    new_df.to_csv(out_path, index=False)

    # # 格式化输出
    # output = new_df.to_string(index=False, float_format='{:.2f}'.format)

    # print(output)


    # out_path = os.path.join(pred_path, "result.json")
    # auto_save_data(scores, out_path)

if __name__ == '__main__':
    Fire(main)
    log_c("evaluation finish")
