from modelzipper.tutils import *
import datasets


all_data_dict, created_data = {}, []

all_data = auto_read_data("/data/zecheng/data/process_wiki_document/one_hop/merged_data/processed_data_2.jsonl")
combine_index = datasets.load_from_disk("/data/zecheng/data/process_wiki_document/one_hop/merged_data/combine_data_hf_2")

for item in all_data:
    all_data_dict[item.pop("id")] = item

created_data = []

for item in combine_index:
    all_ref_ids, chosen_id, rejected_id = item["all_ref_ids"], item["chosen_id"], item["rejected_id"]
    chosen_id, rejected_id = chosen_id, rejected_id
    all_refs = " [DOC] ".join([all_data_dict[i]['context'] for i in all_ref_ids])
    
    question, chosen_ref, rejected_ref = all_data_dict[chosen_id]['question'], all_data_dict[chosen_id]['ref'], all_data_dict[chosen_id]['ref']
    chosen_answer, rejected_answer = all_data_dict[chosen_id]['answer'], all_data_dict[rejected_id]['answer']
    
    question = f"{all_refs} Question: {question}"
    chosen_answer = f"Answer: ###The reference is: {chosen_ref} ###The Answer is: {chosen_answer}"
    rejected_answer = f"Answer: ###The reference is: {rejected_ref} ###The Answer is: {rejected_answer}"
    
    data_item = {"question": question, "chosen_answer": chosen_answer, "rejected_answer": rejected_answer}
    created_data.append(data_item)

keys = created_data[0].keys()
data_dict = {key: [dic[key] for dic in created_data] for key in keys}
dataset = datasets.Dataset.from_dict(data_dict)
dataset.save_to_disk("/data/zecheng/data/process_wiki_document/one_hop/hf_data_simpo")

# auto_save_data(created_data, "/data/zecheng/data/process_wiki_document/one_hop/merged_data/processed_data_3.jsonl")