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
        self.vocab = load_vocab(config["vocab_path"])  # 加载词汇表
        self.config["vocab_size"] = len(self.vocab)  # 更新配置中的词汇表大小
        self.schema = load_schema(config["schema_path"])  # 加载schema
        self.train_data_size = config["epoch_data_size"]  # 设定采样数量
        self.data_type = None  # 标识加载的是训练集还是测试集 "train" or "test"
        self.load()  # 加载数据

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
                        input_id = self.encode_sentence(question)  # 编码句子
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)  # 按schema分类存储
                # 加载测试集
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)  # 编码句子
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])  # 存储测试数据
        return

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":  # 根据词汇表类型选择分词方式
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))  # 获取词的索引，不存在则用[UNK]
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))  # 获取字的索引，不存在则用[UNK]
        input_id = self.padding(input_id)  # 填充或截断
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]  # 截断
        input_id += [0] * (self.config["max_length"] - len(input_id))  # 填充
        return input_id

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]  # 返回训练集采样数量
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)  # 返回测试集数据长度

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample()  # 随机生成一个训练样本
        else:
            return self.data[index]  # 返回测试集指定样本

    # 随机生成3元组样本，2正1负
    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        # 先选定两个意图，之后从第一个意图中取2个问题，第二个意图中取一个问题
        p, n = random.sample(standard_question_index, 2)
        # 如果某个意图下刚好只有一条问题，那只能两个正样本用一样的
        if len(self.knwb[p]) == 1:
            s1 = s2 = self.knwb[p][0]
        # 这是一般情况
        else:
            s1, s2 = random.sample(self.knwb[p], 2)
        # 随机一个负样本
        s3 = random.choice(self.knwb[n])
        # 前2个相似，后1个不相似
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
