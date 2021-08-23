from typing import Dict

from numpy.lib.histograms import _ravel_and_check_weights
import datasets
from datasets import load_dataset, Dataset
import transformers
from transformers import AutoTokenizer
from copy import deepcopy
import pandas as pd

class DataFilter(object):
    '''使用pandas来完成数据提取（复杂度有点感人，但只能选择相信pandas内置函数的速度了）'''
    def __init__(self, base_dataset, cfg):
        self.base_dataset_pd = pd.DataFrame(base_dataset['train'][:])
        # 分词器
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['MODEL_NAME'])
        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast)

    def __call__(self, example) :
        # 一个布尔数组
        # 应该不会重复。。吧
        # TODO
        row = self.base_dataset_pd['doc_id'] == example['doc_id'] 

        tokenized_inputs = self.tokenizer(self.base_dataset_pd.loc[row].iloc[0]['text'],padding="max_length", truncation=True)
        tokenized_inputs["label"] = example['immigration']

        self.base_dataset_pd.drop(self.base_dataset_pd.loc[row].index)
            
        return tokenized_inputs

    def filterRawData(self,example):
        tokenized_inputs = self.tokenizer(example['text'],padding="max_length" , truncation=True)
        tokenized_inputs["label"] = 0
        return tokenized_inputs

    def generate_remain_dataset(self):
        dataset = Dataset.from_pandas(self.base_dataset_pd)
        dataset = dataset.map(self.filterRawData, batched=False)
        return dataset


class DataLoader(object):
    def __init__(self, cfg, mode = 'train'):
        dataset = load_dataset('csv', data_files=['./data/dat_speeches_043114_immi_h_ascii_07212021.csv', 
                                                            './data/dat_speeches_043114_immi_s_ascii_07202021.csv'])
        dataset_label=load_dataset('csv', data_files=['./data/hand_coding_task_house_1000_07162021_lite.csv',
                                                            './data/hand_coding_task_senate_1000_07032021_lite.csv'])
        # 清洗，合并数据集
        filter = DataFilter(dataset, cfg)
        raw_datasets = dataset_label.map(filter, batched=False)
        self.train = None
        self.test = None
        
        if (mode == 'train'):
            raw_datasets.shuffle()
            self.generate_5fold_cross_validation(raw_datasets)
        if (mode == 'inference'):
            self.train, self.test = raw_datasets['train'], None
        
    # 对数据集进行拆分
    def generate_5fold_cross_validation(self, raw_datasets):
        self.train, self.test = [None]*5, [None]*5
        sub_dataset = [raw_datasets['train'].shard(num_shards =5 , index = i) for i in range(5)]
        for i in range(5):
            self.train[i] = datasets.concatenate_datasets([sub_dataset[j] for j in range(5) if j!=i])
            self.test[i] = sub_dataset[i]
        

def test():
    from config import cfg
    dataset = DataLoader(cfg, mode = 'train')
    print(dataset.train)
    print(dataset.test)


if __name__=='__main__':
    test()