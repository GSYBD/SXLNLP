
# -*- coding: utf-8 -*-
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import os

"""
数据加载
"""

import json

class DataGenerator(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        bert_model_path = os.path.abspath(config["bert_path"])
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.schema = self.load_schema(config["schema_path"])  # 确保这个方法存在
        self.data = self.load_data()

    def load_schema(self, schema_path):
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        return schema

    def load_data(self):
        # 加载数据的逻辑
        pass


    def load_data(self):
        data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                encoding = self.tokenizer("".join(sentence), truncation=True, padding='max_length', max_length=self.config["max_length"], return_tensors="pt")
                input_ids = encoding['input_ids'].squeeze(0)
                attention_mask = encoding['attention_mask'].squeeze(0)
                labels = self.padding(labels, -1)
                data.append([input_ids, attention_mask, torch.LongTensor(labels)])
        return data

    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
