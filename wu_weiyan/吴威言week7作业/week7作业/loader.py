# -*- coding: utf-8 -*-

import numpy as np
import tokenizers
import pandas as pd
import torch
import os
import re
import json
from torch.utils.data import DataLoader,Dataset
from transformers import BertTokenizer
import config

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '正面评价', 1: '负面评价'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config['class_nums'] = len(self.index_to_label)
        if self.config['model_type'] == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(config['pretrained_model_path'])
        self.vocab = load_vocab(config['vocab_path'])
        self.config['vocab_size'] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []

        df = pd.read_csv(self.path)
        for i in range(len(df)):
            label = df.iloc[i]['label']
            review = df.iloc[i]['review']

            if self.config['model_type'] == 'bert':
                input_id = self.tokenizer.encode(review, max_length=self.config['max_length'])
            else:
                input_id = self.encode_sentence(review)

            input_id = torch.LongTensor(input_id)
            index = torch.LongTensor([label])
            self.data.append([input_id, index])

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    dg = DataGenerator("../data/文本分类练习.csv", config.Config)
    print(dg[1])