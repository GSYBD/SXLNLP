"""
数据加载类
"""
import json
import torch
import csv
from transformers import BertTokenizer
from torch.utils.data import DataLoader
class DataGenerator:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.config['class_num'] = 2
        self.vocab = load_vocab(config['vocab_path'])
        self.config['vocab_size'] = len(self.vocab)
        if self.config['use_bert']:
            self.tokenizer = BertTokenizer.from_pretrained(self.config['pretrain_model_path'])
        self.load_data()
    def load_data(self):
        self.data = []
        with open(self.data_path,encoding='utf8') as f:
            reader = csv.reader(f)
            for index, row in enumerate(reader):
                if index == 0:
                    continue
                label, review = row
                label_index = torch.LongTensor([int(label)])
                if self.config['use_bert']:
                    input_id = self.tokenizer.encode(review, max_length=self.config['max_length'], pad_to_max_length=True)
                else:
                    input_id = self.sentence_encode(review)
                input_id = torch.LongTensor(input_id)
                self.data.append([input_id, label_index])


    def sentence_encode(self, sentence):
        sequence = [self.vocab.get(c, self.vocab['[UNK]']) for c in sentence]
        return self.padding(sequence)
    def padding(self, sequence):
        sequence = sequence[:self.config['max_length']]
        sequence += [0] * (self.config['max_length'] - len(sequence))
        return sequence
    def __getitem__(self, item):
        return self.data[item]
    def __len__(self):
        return len(self.data)
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict
def load_data(data_path, config, shuffle=True):
    ds = DataGenerator(data_path, config)
    dl = DataLoader(ds, batch_size=config['batch_size'], shuffle=shuffle)
    return dl

if __name__ == '__main__':
    from config import Config
    ds = DataGenerator(Config['train_data_path'], Config)
    print(ds[0])