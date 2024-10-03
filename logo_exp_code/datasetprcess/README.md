## 关键路径获取的脚本
## 根据关键路径生成的脚本

## 关键路径生成结果评测的脚本

下面这段代码处理2的生成结果, 会走doubao API对生成结果进行评测
```shell
python vllm_generation/post_process_critical_paths.py
```

1. 评测标准: -1表示标注失败,0表示模型推理结果和label完全不一致, 2表示pred1比pred2好,且pred1和label的标准一致, 1表示pred1和pred2可能同时对/错,但是pred1和label一致.
2. 生成结果的数据格式: 都是存储成datasets.Dataset的格式,key分别是: 
```python
partial_result['combined_question'].append(item1['question'])
partial_result['label'].append(item1['label'])
partial_result['final_answer'].append(item1['predict'])
partial_result['prefix_a'].append(item2['predict'])
partial_result['siffix_a'].append(item3['predict'])
partial_result['all_ref_text'].append(item1['context_lst'])  # 这里是一个List[str]
partial_result['judge_scores'].append(judge_score)
partial_result['judger_preds'].append(judge_res_str)
```

生成结果在`/data/zecheng/data/llama3-80k-train-data/long-llm-score`

## 根据关键路径生成训练数据的脚本

1. 根据上述的规则,优先走分数为2的结果,看看有多少条数据

2. 数据生成脚本
```shell
python iclr2025/datasetprcess/build_training_data/hpsw.py # 生成skip-wise的训练数据结果
```

3. 数据格式
```python
concatenated_batch['chosen'] = {
    'input_ids': torch.concat([q_input_ids, cont_qa['chosen_answer'][0], torch.zeros(chosen_padding_size, dtype=torch.int)], dim=0), 
    'attention_mask': torch.concat([q_attention_mask, cont_qa['chosen_answer'][1], torch.zeros(chosen_padding_size, dtype=torch.int)], dim=0), 
    'position_ids': torch.concat([q_position_ids, cho_a_padded_pos_ids, torch.zeros(chosen_padding_size, dtype=torch.int)], dim=0), 
    'labels': torch.concat([referece_question_labels, cont_qa['chosen_answer'][3], torch.full((chosen_padding_size,), -100, dtype=torch.int)], dim=0) 
}
concatenated_batch['reject_1'] = {
    'input_ids': torch.concat([q_input_ids, cont_qa['rejected_answer1'][0], torch.zeros(reject_1_padding_size, dtype=torch.int)], dim=0),
    'attention_mask': torch.concat([q_attention_mask, cont_qa['rejected_answer1'][1], torch.zeros(reject_1_padding_size, dtype=torch.int)], dim=0),
    'position_ids': torch.concat([q_position_ids, rej_a1_padded_pos_ids, torch.zeros(reject_1_padding_size, dtype=torch.int)], dim=0),
    'labels': torch.concat([referece_question_labels, cont_qa['rejected_answer1'][3], torch.full((reject_1_padding_size,), -100, dtype=torch.int)], dim=0)
}
concatenated_batch['reject_2'] = {
    'input_ids': torch.concat([q_input_ids, cont_qa['rejected_answer2'][0], torch.zeros(reject_2_padding_size, dtype=torch.int)], dim=0), 
    'attention_mask': torch.concat([q_attention_mask, cont_qa['rejected_answer2'][1], torch.zeros(reject_2_padding_size, dtype=torch.int)], dim=0),
    'position_ids': torch.concat([q_position_ids, rej_a2_padded_pos_ids, torch.zeros(reject_2_padding_size, dtype=torch.int)], dim=0),  
    'labels': torch.concat([referece_question_labels, cont_qa['rejected_answer2'][3], torch.full((reject_2_padding_size,), -100, dtype=torch.int)], dim=0)
}
```