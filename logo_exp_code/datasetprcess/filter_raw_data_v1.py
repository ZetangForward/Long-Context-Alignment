from typing import List, Set, Tuple, Dict, Optional
import spacy
from modelzipper.tutils import *
from tqdm.contrib.concurrent import process_map

class DataFilter:

    def __init__(self, dataset_dir, dataset_names, model_id, output_path, save_chunk_nums=12, chunk_size=512, min_ne_overlap=3, num_process=1):
        self.dataset_dir = dataset_dir
        self.dataset_names = dataset_names
        self.model_id = model_id
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.save_chunk_nums = save_chunk_nums
        self.min_ne_overlap = min_ne_overlap
        self.num_process = num_process
        self.init_filter()

    def init_filter(self):
        print(f'Initializing filter with model {self.model_id}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        print(f'Loading datasets from {self.dataset_dir}')
        self.datasets = {}
        with tqdm(total=len(self.dataset_names), desc='Loading datasets') as pbar:
            for dataset_name in self.dataset_names:
                dataset_path = f'{self.dataset_dir}/{dataset_name}'
                self.datasets[dataset_name] = auto_read_data(dataset_path)
                pbar.update(1)
        print('Loading spacy model')
        self.spacy_model = spacy.load("en_core_web_md")


    def chunk_text(self, s: str, chunk_size: int=1024) -> List[str]:
        '''
        chunk text according to the chunk_size and tokenizer
        return: List[str]
        '''
        tok_s = self.tokenizer(s, return_tensors='pt').input_ids[0]
        tok_chunks = [tok_s[i:i+chunk_size] for i in range(0, len(tok_s), chunk_size)]
        return [self.tokenizer.decode(chunk) for chunk in tok_chunks]


    def get_ne_from_s(self, s) -> Set[str]:
        '''Example: 
        s="What was the cause of the teenager's death in the motorcycle accident? The teenager, John Lister, was killed in a high-speed crash."
        verb = ['kill']
        ents = ['John Lister']
        none = ['cause', 'teenager', 'death', 'motorcycle', 'accident', 'teenager', 'speed', 'crash']
        '''
        doc = self.spacy_model(s)
        verb = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        ents = [chunk.text for chunk in doc.ents]
        none = [token.lemma_ for token in doc if token.pos_ == "NOUN"]
        return set(verb + ents + none)


    def judge_ne_overlap(self, critical_paths: List[Dict], min_ne_overlap: int = 3) -> bool:
        '''
        Judge if the critical paths have enough overlap
        '''
        all_overlaps = [path['overlap'] for path in critical_paths]
        if min(all_overlaps) >= min_ne_overlap:
            return True
        return False


    def process_sample(self, q: str, a: str, s: str, chunk_size: int=512, save_chunk_nums: int=12, min_ne_overlap: int=3) -> Optional[Dict]:
        '''
        Extract Critical Paths from s, according to the question q and answer a
        '''
        q_ne, a_ne = self.get_ne_from_s(q), self.get_ne_from_s(a)
        s_chunks = self.chunk_text(s, chunk_size)
        union_nes = q_ne.union(a_ne)
        critical_chunks = []
        for i, chunk in enumerate(s_chunks):
            chunk_ne = self.get_ne_from_s(chunk)
            overlap = len(union_nes.intersection(chunk_ne))
            critical_chunks.append({'chunk_id': i, 'chunk': chunk, 'overlap': overlap}) 
        sorted_critical_chunks = sorted(critical_chunks, key=lambda x: x['overlap'], reverse=True)[:save_chunk_nums]
        final_critical_chunks = []
        if self.judge_ne_overlap(sorted_critical_chunks, min_ne_overlap):
            # final_critical_chunks = sorted(sorted_critical_chunks, key=lambda x: x['chunk_id'])
            # final_critical_chunks = [chunk['chunk'] for chunk in final_critical_chunks]
            return {'question': q, 'answer': a, 'context': sorted_critical_chunks}
        return None


    def mp_process_dataset(self, content: List[Dict]) -> List[Dict]:
        '''
        Process a dataset using multiprocessing with tqdm progress bar
        '''
        results = process_map(
            self.process_item,
            content,
            max_workers=self.num_process,
            desc="Processing dataset",
            chunksize=1
        )
        
        processed_data = []
        for result in results:
            processed_data.extend(result)
        
        return processed_data
    
    def process_item(self, item):
        item = item['conversations']
        context = '\n\n'.join(item[0]['content'].split('\n\n')[:-1]).strip()
        all_questions, all_answers = item[2::2], item[3::2]
        processed_items = []
        # with tqdm(total=len(all_questions), desc=f'Processing one single sample, which has {len(all_questions)} queries ...') as pbar:
        for q, a in zip(all_questions, all_answers):
            processed_item = self.process_sample(q['content'], a['content'], context, self.chunk_size, self.save_chunk_nums, self.min_ne_overlap)
            if processed_item:
                processed_items.append(processed_item)
                # pbar.update(1)
        return processed_items

    def process_dataset(self, content: List[Dict]) -> List[Dict]:
        '''
        Process a dataset
        '''
        processed_data = []
        with tqdm(total=len(content), desc='process datasets ...') as pbar1:
            for item in content:
                item = item['conversations']
                context = '\n\n'.join(item[0]['content'].split('\n\n')[:-1]).strip()
                all_questions, all_answers = item[2::2], item[3::2]
                
                with tqdm(total=len(all_questions), desc=f'Processing one single sample, which has {len(all_questions)} queries ...') as pbar2:
                    for q, a in zip(all_questions, all_answers):
                        processed_item = self.process_sample(q['content'], a['content'], context, self.chunk_size, self.save_chunk_nums, self.min_ne_overlap)
                        if processed_item:
                            processed_data.append(processed_item)
                        pbar2.update(1)
                
                pbar1.update(1)
        return processed_data
    

    def begin_process(self):
        '''
        Begin to process
        '''
        # all_processed_data = {}
        for dataset_name in self.dataset_names:
            print(f'Processing dataset {dataset_name}, which has {len(self.datasets[dataset_name])} samples.')
            processed_data = self.mp_process_dataset(self.datasets[dataset_name])
            save_path = f'{self.output_path}/{dataset_name.split("/")[-1].split(".")[0]}.jsonl'
            auto_save_data(processed_data, save_path)
            # all_processed_data[dataset_name] = processed_data
        # return all_processed_data



if __name__ == '__main__':
    dataset_dir = '/data/zecheng/data/llama3-80k-train-data/long-llm'
    dataset_names = [
        'gpt/one_detail_book.train.64K.json',
        'gpt/one_detail_paper.train.64K.json',
        'gpt/multi_detail_book.train.json',
        'gpt/multi_detail_paper_short.train.json',
        'gpt/multi_detail_paper_long.train.json',
        'gpt/bio_book.train.json',
        # 'longalpaca/train.json',
        # 'redpajama/train.json[5000]',
    ]

    data_filter = DataFilter(
        dataset_dir, dataset_names, 
        model_id='/data/zecheng/hf_models/Llama-3-8B-Instruct-80K-QLoRA-Merged',
        output_path='/data/zecheng/data/llama3-80k-train-data/long-llm-filtered',
        save_chunk_nums=12, chunk_size=512, min_ne_overlap=3, num_process=128,
    )

    data_filter.begin_process()
    # auto_save_data(process_res, '/data/zecheng/data/llama3-80k-train-data/long-llm-filtered/rush.pkl')
