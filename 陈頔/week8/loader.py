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
        self.load()

    def load(self):
        # 保持测试集数据[input_id/questions, target/label]
        self.data = []
        # 保存训练集序列化数据 {label1:[input_id1,input_id2...],label2:[input_id1,input_id2...]...}
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf8") as f:
            for line in f:  
                # 将每行数据从JSON字符串转换为Python对象  
                line = json.loads(line)  
                
                # 加载训练集数据  
                if isinstance(line, dict):  
                    self.data_type = "train"  # 标记当前处理的是训练集数据  
                    questions = line["questions"]  # 获取问题列表  
                    label = line["target"]  # 获取标签  
                    for question in questions:  
                        # 对每个问题进行编码  
                        input_id = self.encode_sentence(question)  
                        # 将编码后的数据转换为PyTorch的LongTensor  
                        input_id = torch.LongTensor(input_id)  
                        # 根据标签在self.schema中的映射，将编码后的数据添加到对应的列表中  
                        self.knwb[self.schema[label]].append(input_id)  
                
                # 加载测试集数据  
                else:  
                    self.data_type = "test"  # 标记当前处理的是测试集数据  
                    # 断言当前行数据为列表类型，包含问题和标签  
                    assert isinstance(line, list)  
                    question, label = line  # 解包列表，获取问题和标签  
                    # 对问题进行编码  
                    input_id = self.encode_sentence(question)  
                    # 将编码后的数据转换为PyTorch的LongTensor  
                    input_id = torch.LongTensor(input_id)  
                    # 将标签转换为PyTorch的LongTensor，并映射为self.schema中的索引  
                    label_index = torch.LongTensor([self.schema[label]])  
                    # 将编码后的问题和标签索引添加到测试集数据列表中
                    # 测试和训练的逻辑不一样
                    # 需要判断输入和真实target/label是否一致，一致才算测试集通过  
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
            return self.random_train_sample() #随机生成一个训练样本
        else:
            return self.data[index]

    #依照一定概率生成负样本或正样本
    #负样本从随机两个不同的标准问题中各随机选取一个
    #正样本从随机一个标准问题中随机选取两个
    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        if self.config["loss_type"] == "cosine_embedding":
            #随机正样本
            if random.random() <= self.config["positive_sample_rate"]:
                p = random.choice(standard_question_index)
                #如果选取到的标准问下不足两个问题，则无法选取，所以重新随机一次
                if len(self.knwb[p]) < 2:
                    return self.random_train_sample()
                else:
                    s1, s2 = random.sample(self.knwb[p], 2)
                    return [s1, s2, torch.LongTensor([1])]
            #随机负样本
            else:
                p, n = random.sample(standard_question_index, 2)
                s1 = random.choice(self.knwb[p])
                s2 = random.choice(self.knwb[n])
                return [s1, s2, torch.LongTensor([-1])]
        elif self.config["loss_type"] == "cosine_triplet":
            # random.sample(population, k)：这个函数从指定的序列population中随机获取k个不重复的元素
            p, n = random.sample(standard_question_index, 2)
            # 如果选取到的标准问下不足两个问题，则重新随机一次
            if len(self.knwb[p]) < 2:
                return self.random_train_sample()
            else:
                # 随机选取两个问题作为正样本
                # 从self.knwb字典中键为p的值中随机选择两个不重复的元素,也就是在一个标准问下随机选择两个相似问
                a, p = random.sample(self.knwb[p], 2) # a是anchor，p是positive
                n = random.choice(self.knwb[n]) # n是negative，从另一个标准问下随机选择一个不相似问
                return [a, p, n]


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
    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])
