import torch
import numpy as np
import json


from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class DataGenerator(object):
    def __init__(self,data_path, config:dict):
        self.config = config
        self.path = data_path
        
        self.index_to_label = {0: '好评', 1: '差评'}
        
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
    
        self.config['class_num'] = len(self.index_to_label)
        
        if self.config['model_type'] == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(self.config['pretrain_model_path'])
        self.vocab = load_vocab(self.config['vocab_path'])
        self.config['vocab_size'] = len(self.vocab)
        self.config["class_num"] = len(self.index_to_label)
        
        self.load() # 加载self.data
        
    def load(self):
        '''
         读取数据集，并转换成训练数据
        
        '''
        self.data = []
        with open(self.path, encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                # print("line = \n",line)
                
                label = int(line['label'])
                content = line['review']
                
                input_id = None
                if self.config['model_type'] == 'bert':
                    input_id = self.tokenizer.encode(content, max_length = self.config['max_length'], padding = 'max_length')
                else:
                    input_id = self.encode_sentence(content)
                
                input_id = self.padding(input_id)
                
                input_id = torch.LongTensor(input_id)
                label = torch.LongTensor([label])
                self.data.append([input_id, label])
                
        return 
            
    def encode_sentence(self, text):
        input_id = []
        
        for char in text:
            input_id.append(self.vocab.get(char,self.vocab['[UNK]']))
        return input_id
    
    def padding(self, input_id):
        
        # 截断, 比max_len长才会截断，否则执行无效
        input_id = input_id[:self.config['max_length']]
        
        # padding
        input_id += [0]*(self.config['max_length']-len(input_id))

        return input_id
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]



def load_vocab(path):
    vocab = {}
    with open(path, 'r', encoding='utf8') as f:
        for index,line in enumerate(f):
            line = line.strip()
            vocab[line] = index 
    return vocab



# 封装DataLoader
def load_data(path, config, shuffle=True):
    dg = DataGenerator(path, config)
    
    dataloader = DataLoader(dg,batch_size=config["batch_size"], shuffle=shuffle)


    return dataloader





if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("data/valid_tag_news.json", Config)
    print(dg[1])

