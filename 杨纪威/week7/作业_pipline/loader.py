# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import csv
import sys
from io import StringIO
"""
数据加载
"""
# 从config配置表里面取值
from config import Config
print(Config["model_path"])
index_to_label = {1: '好评',0:'差评'}
label_to_index = {}
for x ,y in index_to_label.items():
    label_to_index[y] = x
# print(label_to_index)


class DataGeneraor:
    def __init__(self,data_path,config):
        self.config = config
        self.path = data_path
        self.index_to_label = {1: '好评',0:'差评'}
        label_to_index = {}
        for x ,y in self.index_to_label.items():
            label_to_index[y] = x
        self.label_to_index = label_to_index
        self.config['class_num'] = len(self.index_to_label)
        # print(self.config['class_num'])
        self.tokenizer = BertTokenizer.from_pretrained(config['pretrain_model_path'])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()
    def load(self):
        self.data = []
        # label_list =[]

        with open(self.path, encoding='utf8') as f:
            csv_reader = csv.DictReader(f)  # 这个方法通常用于读取包含标题行的CSV文件，它会将每一行数据读取为一个字典，字典的键是标题行中的列名，值是对应列的数值。
            # 遍历每行数据
            for row in csv_reader:
                # 提取"label"列的数值并添加到列表中
                label_list = row['label']
                # print("label_list",label_list)
                if self.config['model_type'] == 'bert':
                    input_id = self.tokenizer.encode(row['review'],max_length=self.config["max_length"], pad_to_max_length=True)
                    # print("input_id:",input_id)
                else:
                    input_id = self.encode_sentence(row['review'])
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([int(label_list)])
                self.data.append([input_id, label_index])
        return
    def encode_sentence(self,text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self,input_id):
        input_id = input_id[:self.config["max_length"]]  # 将输入的 input_id 列表截取至最大长度 self.config["max_length"]。如果列表长度超过了最大长度，则截取到最大长度；如果列表长度小于最大长度，则保持不变。
        input_id += [0] * (self.config["max_length"] - len(input_id))  # 计算需要填充的数量，即最大长度减去当前列表的长度。然后将这个数量的 0 添加到列表的末尾，以达到最大长度。
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_data(data_path,config,shuffle = True):
    dg = DataGeneraor(data_path,config)
    dl = DataLoader(dg,batch_size=config["batch_size"],shuffle=shuffle)
    return dl

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path,encoding="utf-8") as f:
        # lines = f.readlines()
        for index ,line in enumerate(f):
            token = line.strip()
            # print(line)
            token_dict[token] = index + 1
    return token_dict
vocab_dict = load_vocab(Config["vocab_path"])
# 设置标准输出的编码为UTF-8
# sys.stdout.reconfigure(encoding='utf-8')
# # 打印返回的词汇表字典
# print(vocab_dict)

if __name__ == '__main__':
    dg = DataGeneraor('../作业_data/valid.csv',Config)
    print("dg",dg[1000])
    # for data in dg:
    #     print(data[1])

