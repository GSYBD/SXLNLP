# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.sentences = []
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")  # 两个换行是一句话
            for segment in segments:
                sentenece = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()  # 空格分开，前一个是字，后一个是标签
                    sentenece.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentenece))
                # 意义对应的关系
                # print("sentenece",sentenece)
                # print("labels",labels)
                # 去掉前缀 ，还没有调试成功
                input_ids = self.tokenizer.encode(sentenece, max_length=self.config["max_length"], truncation=True,
                                                  padding='max_length')  # ,add_special_tokens=False
                # labels_index = self.padding(labels, -1)
                # 转化成 深度学习可以识别的张量
                input_ids = self.encode_sentence(sentenece)
                input_ids = torch.LongTensor(input_ids)
                labels_index = torch.LongTensor(self.padding(labels, -1))
                self.data.append([input_ids, labels_index])
        return

    def encode_sentence_bert(self, text, padding=True):
        x = self.tokenizer.encode(text, padding='max_length', max_length=self.config["max_length"],truncation=True)
        if len(x) > 100:
            print(text)
        return x


    def encode_sentence(self, text, padding=True):
        if self.config["model_type"]=='bert':
            return self.encode_sentence_bert(text,padding=True)

        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    # bert 在embedding 时候会加上前后token,所以要特殊处理. 其实可以在使用add_special_tokens=False 去掉,就不需要特殊处理
    def padding_bert(self, input_id, pad_token=0):
        x = [8]
        max_len = self.config["max_length"] -2
        input_id = input_id[:max_len]
        input_id += [pad_token] * (max_len - len(input_id))
        x += input_id
        x += [-1]
        return x



    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        if self.config["model_type"]=='bert':
            return self.padding_bert(input_id,pad_token)

        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)
#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator(Config['train_data_path'], Config)
    print(dg[0])

