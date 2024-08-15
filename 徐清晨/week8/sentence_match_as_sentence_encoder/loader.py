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
        """
        self.knwb: 每个类别下都有哪些句子，将这些句子转为词向量
        defaultdict(<class 'list'>, {2: [tensor([4270,  157,  164, 1548, 2769, 2685, 3761,  669,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0]), tensor([ 540, 2626,  173,  543,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0]), tensor([ 540, 2626, 2799,  434,  173,  543,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0]), tensor([1548, 1798,  513,  157,  183,  361, 1457, 1880, 1427,  197,  858,  223,
           0,    0,    0,    0,    0,    0,    0,    0]), tensor([2142, 1289,  173,  543,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0]), tensor([1548, 1498,  738, 3766, 2685,  361,  173,  543,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0]), tensor([1183, 1880, 4013,  183, 2685, 3761,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0]), tensor([1548, 1498, 3694,  183, 1569, 1949,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0]), tensor([4087,  183, 2685, 3761,    0,    0,    0,    0,    0,    0,    0,    0,

        :param data_path:
        :param config:
        """
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])

        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])

        self.train_data_size = config["epoch_data_size"]  # 由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.data_type = None  # 用来标识加载的是训练集还是测试集 "train" or "test"
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)

                # 加载训练集
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)

                # 加载测试集
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    # print(input_id)
                    # exit()
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])
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

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == "train":
            # 因为这里随机生成无限多
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample()  # 随机生成一个训练样本
        else:
            return self.data[index]

    # 负样本从随机两个不同的标准问题中随机选取一个
    # 正样本从随机一个标准问题中随机选取两个
    def random_train_sample(self):
        """
        生成三个样本，两个正的一个负的

        :return:
        """
        # standard_question_index: 所有分类的索引
        standard_question_index = list(self.knwb.keys())
        p, n = random.sample(standard_question_index, 2)
        if len(self.knwb[p]) < 2:
            # 如果选取到的正样本不足两个，就重新来一次
            return self.random_train_sample()

        s1, s2 = random.sample(self.knwb[p], 2)
        s3 = random.choice(self.knwb[n])
        return [s1, s2, s3]



# 加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


# 加载schema
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])
