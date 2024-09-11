# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import random
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.data = []
        self.labels = []
        self.train_data = []
        self.test_data = []
        self.load()

    def load(self):
        with open(self.path, encoding="utf8") as f:
            for line_number, line in enumerate(f, start=1):
                if line_number == 1:
                    continue
                label = int(line.strip()[:1])
                if label not in self.labels:
                    self.labels.append(label)
                review = line.strip()[2:]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(review, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(review)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        random.shuffle(self.data)
        self.train_data = self.data[:round(len(self.data) * 0.7)]
        self.test_data = self.data[round(len(self.data) * 0.7):]
        self.config['class_num'] = len(self.labels)
        return

    def encode_sentence(self, text):
        input_id = []
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
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=False):
    dg = DataGenerator(data_path, config)
    dl = (
        DataLoader(dg.train_data, batch_size=config["batch_size"], shuffle=shuffle),
        DataLoader(dg.test_data, batch_size=config["batch_size"], shuffle=shuffle),
    )
    return dl


if __name__ == "__main__":
    from config import Config
    dg = DataGenerator(r"D:\书\八斗精品班\第七周 文本分类问题\week7 文本分类问题\文本分类练习.csv", Config)
    print(dg[1])
