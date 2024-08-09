# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"] #由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.data_type = None  #用来标识加载的是训练集还是测试集 "train" or "test"
        self.labels = {}
        self.load()
    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                #加载训练集
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    self.labels[self.schema[label]] = label
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                #加载测试集
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])
        # print("self.labels",self.labels)
        return

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
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
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            # 在训练模式下，不需要 index，因为 random_train_sample 自己处理随机性
            return self.random_train_sample()
        else:
            # 在测试模式下，使用 index 来索引测试数据
            return self.data[index]
    #依照一定概率“生成”负样本或正样本
    #负样本从随机两个不同的标准问题中各随机选取一个
    #正样本从随机一个标准问题中随机选取两个
    # 一般会设置小数，大于这个小数是正样本，小于这个小数 时候负样本，控制正负样本的比例

    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())  # 得到有多少个 label
        # print("standard_question_index",standard_question_index)
        #随机正样本
        if random.random() <= self.config["positive_sample_rate"]:
            p, n = random.sample(standard_question_index, 2)
            questions1 = self.knwb[p]
            questions2 = self.knwb[n]
            #如果选取到的标准问下不足两个问题，则无法选取，所以重新随机一次
            if len(questions1) < 2:
                return self.random_train_sample()
            else:
                s1, s2 = random.sample(questions1, 2)
                # s3 = random.sample(questions2, 1)
                s3 = random.choice(questions2)  # 直接选择单个样本，而不是列表
                # print( "正样本：",[s1, s2, s3])
                return [s1, s2, s3]

        #随机负样本
        else:
            a, p, n = random.sample(standard_question_index, 3) # 先选择两标签，从标签里面选择
            s1 = random.choice(self.knwb[a])  # choice() 函数从一个与某个键 p 相关联的列表中随机选择一个元素
            s2 = random.choice(self.knwb[p])
            s3 =random.choice(self.knwb[n])
            # print("负样本数据正常")
            return [s1, s2, s3]

#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#加载schema
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../data/train.json", Config)
    print(dg[1])