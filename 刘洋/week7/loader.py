# -*- coding: utf-8 -*-

import json
import re
import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            csv_file= csv.reader(f)
            next(csv_file)
            for line in csv_file:
                csv_key = int(line[0])
                csv_value= line[1]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(csv_value, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(csv_value)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([csv_key])
                self.data.append([input_id, label_index])
        return
#
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
     # 添加未知字符的占位符
    token_dict['[UNK]'] = len(token_dict) + 1  # 将'[UNK]'添加到词汇表的末尾，并分配一个索引
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(vocab_path, config, shuffle=True):
    dg = DataGenerator(vocab_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator(r"D:\刘洋\刘洋(个人)\2024大数据算法学习\学习资料\第七周 文本分类问题\week7 文本分类问题\文本分类练习.csv", Config)
    print(dg[1])
