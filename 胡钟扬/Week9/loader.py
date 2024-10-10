import re
import json
import torch
import os
import jieba
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertTokenizer
from collections import defaultdict    


class DataGenerator(Dataset):
    
    def __init__(self, config, data_path):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config['vocab_size'] = len(self.vocab)
        
        self.sentences = [] # 用来存储数据集中的所有句子
        
        # 加载NER任务的专属vocab
        self.schema = self.load_schema(config['schema_path'])
        
        
        self.load()
    
    
    def load(self):
        self.data = []
        
        with open(self.path, encoding='utf8') as f:
            batches = f.read().split("\n\n")
            
            for batch in batches:
                sentence = [] # 收集每个batch中的所有token
                labels = []
                
                for example in batch.split("\n"):
                    if example.strip() == "":
                        continue
                    char, label = example.split() # 分离 x,y
                    sentence.append(char)
                    labels.append(self.schema[label])
                
                self.sentences.append("".join(sentence))
                input_ids = self.encode_sentence(sentence) # 默认padding, pad_token=0
                labels = self.padding(labels, -1)  # pad_token = -1
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
                
        return

    
    
    
    def encode_sentence(self, text, padding = True):
        input_id = []
        
        for word in text:
            input_id.append(self.vocab.get(word,self.vocab['[UNK]']))
        
        if padding:
            return self.padding(input_id)
        else:
            return input_id
    
    
    def padding(self, input_id, pad_token=0):
        # padding
        input_id += [pad_token]*(self.config['max_length']-len(input_id))
        
        # truncate
        return input_id[:self.config['max_length']]
    
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self,index):
        return self.data[index]
    
    
    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)
        

    
class DataGeneratorforSentence:
    '''
        This dataset is only for the whole sentence NER task
    '''
    def __init__(self, data_path, config, logger): 
        self.path = data_path
        self.logger = logger
        self.config = config
        self.max_length = config['max_length']
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["bert_path"], add_special_tokens=True)
        self.label_map = {"B":0, "I":1, "O":2}
        self.load()
    
    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for segment in f.read().split("\n\n"): # 处理每一段text
                if segment.strip()=="" or "\n" not in segment:
                    continue
                self.prepare_data(segment)
        
        return 
            

    
    def prepare_data(self, segment):
        '''
                处理一段话(1个batch)，把它让分割成很多句话，以及他们对应的标签
                
                self.data.append([
                    torch.LongTensor(segment_input_ids),
                    torch.LongTensor(segment_attention_mask),
                    torch.LongTensor(labels)
                ])
                
                return
        '''
        segment_input_ids = []
        segment_attention_mask = []
        labels = []
        
        for line in segment.split("\n"):
            line = line.strip()
            if line == "":
                continue
            
            sentence = line.split('\t')[0]
            label = line.split('\t')[1][0]
            role =line.split('\t')[3]
            
            assert label in self.label_map and label
            assert role in ["Reply", "Review"]
            
            label = self.label_map[label]
            encode = self.tokenizer.encode_plus(sentence, 
                                                max_length = self.config['max_length'],
                                                pad_to_max_length=True,
                                                add_special_tokens=True)
            input_ids = encode['input_ids']
            attention_mask = encode['attention_mask']
            
            segment_input_ids.append(input_ids)
            segment_attention_mask.append(attention_mask)
            
            labels.append(label)
            if len(labels) > self.config["max_sentence"]:
                break
            
        self.data.append([
            torch.LongTensor(segment_input_ids), # (batch_size, seq_len)
            torch.LongTensor(segment_attention_mask), # (batch_size, seq_len)
            torch.LongTensor(labels) 
        ])
        
        return
            
            
            
    
    
    
    def __getitem__(self,index):
        return self.data[index]
    
    
    
    def __len__(self):
        return len(self.data)
    
    
    
def encode_sentence(text, config):
    vocab = load_vocab(config['vocab_path'])
    input_id = []
        
    for word in text:
        input_id.append(vocab.get(word,vocab['[UNK]']))
    
    return input_id


def load_vocab(path):
    vocab_dict = {}
    
    with open(path, encoding="utf8") as f:
        for index, line in enumerate(f):
            vocab_dict[line.strip()] = index+1 # 0 is reserved for padding
    return vocab_dict
    
        
def load_data(data_path, config, shuffle=True):
    '''
     use Pytorch DataLoader to encapsulate dataset
    '''
    
    data_generator = DataGenerator(config, data_path)
    
    data_loader = DataLoader(data_generator, batch_size=config['batch_size'], shuffle=shuffle)

    return data_loader


def load_data_for_sentence(data_path, config, logger):
    data_generator = DataGeneratorforSentence(data_path, config, logger)
    
    return data_generator



# test 
if __name__ == "__main__":
    from config import Config
    dg = DataGenerator(Config, Config['train_data_path'])