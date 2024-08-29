import json

import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader


class DataGenerator:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config['bert_path'])
        self.load()

    def load(self):
        self.data = []
        with open(self.data_path, 'r') as f:
            for line in f:
                json_data = json.loads(line)
                title = json_data['title']
                content = json_data['content']
                title_seq = self.tokenizer.encode(title, add_special_tokens=False, padding='max_length',
                                                  truncation=True, max_length=self.config['txt1_len'] - 1)
                content_seq = self.tokenizer.encode(content, add_special_tokens=False, padding='max_length',
                                                    truncation=True, max_length=self.config['txt2_len'])
                input_seq = title_seq + [103] + content_seq
                # target_seq = self.tokenizer.encode(content)
                target_seq = self.padding(content_seq, self.config['max_len'])
                self.data.append([torch.LongTensor(input_seq), torch.LongTensor(target_seq)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def padding(self, seq, max_length):
        seq = seq[:max_length]
        seq = [-100]*(max_length - len(seq)) + seq
        return seq


def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config['batch_size'], shuffle=shuffle)
    return dl
