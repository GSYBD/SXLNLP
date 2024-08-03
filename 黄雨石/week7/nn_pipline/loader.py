# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):#config是一个字典对象
        self.config = config
        self.path = data_path
        # 分类的标签 一共18个类别
        self.index_to_label = {0: '差评', 1: '好评'}

       #将标签配对为一个字典
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        #增加类别数据的属性
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert": #直接加载与训练模型
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"]) #加载词向量
        self.config["vocab_size"] = len(self.vocab) #计算词的数据
        self.load()


    def load(self):
        self.data = []
        try:
            df = pd.read_csv(self.path)
            for index,row in df.iterrows():
                label = row["label"]  # 得到标签
                review = row["review"]  # 得到输入文本
                if self.config["model_type"] == "bert":#直接叼你用bert的文本的序列话
                    input_id = self.tokenizer.encode(review, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(review)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        except FileNotFoundError:
            print(f"文件 '{self.path}' 不存在")
        return

    def encode_sentence(self, text):#
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"])) #添加输入的序号
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]] #限制长度
        input_id += [0] * (self.config["max_length"] - len(input_id)) #超出补齐
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


#用torch自带的DataLoader类封装数据 ,提供外部函数使用，得到torch的数据格式
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

def coutData(file_path):
    # 打开文件
    # file_path = 'your_file.txt'  # 替换为你的文件路径
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            line_count = sum(1 for line in file)
    except FileNotFoundError:
        print(f"文件 '{file_path}' 不存在")
    else:
        print(f"文件 '{file_path}' 有 {line_count} 行")

def cutdata(file_path):

    # 读取CSV文件
    # file_path = 'your_dataset.csv'  # 替换为你的CSV文件路径
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"文件 '{file_path}' 不存在")
        exit()

    # 获取数据集大小
    total_samples = len(df)
    print(f"总样本数：{total_samples}")

    # 定义训练集和验证集的比例
    train_ratio = 0.8  # 训练集比例为4:1，这里是0.8
    # validation_ratio = 0.2  # 验证集比例为1:1，这里是0.2

    # 根据比例随机拆分数据集
    train_size = int(total_samples * train_ratio)
    validation_size = total_samples - train_size

    # 打印拆分后的数据集大小
    print(f"训练集大小：{train_size}，验证集大小：{validation_size}")

    # 使用numpy生成随机索引
    indices = np.random.permutation(df.index)
    train_indices = indices[:train_size]
    validation_indices = indices[train_size:train_size + validation_size]

    # 根据索引拆分数据集
    train_set = df.loc[train_indices]
    validation_set = df.loc[validation_indices]

    # 输出拆分后的数据集信息
    print(f"训练集样本数：{len(train_set)}")
    print(f"验证集样本数：{len(validation_set)}")

    # 将拆分后的数据集保存为新的CSV文件
    train_set.to_csv('./data/train_set.csv', index=False)
    validation_set.to_csv('./data/validation_set.csv', index=False)

    print("训练集和验证集已保存为 train_set.csv 和 validation_set.csv")


if __name__ == "__main__":
    from config import Config #显然这一部必须加入，配置是作为参数传入
    dg = DataGenerator("./data/train_set.csv", Config)
    print(dg[1])

