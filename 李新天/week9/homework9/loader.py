# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.config["class_num"] = len(self.schema)
        self.max_length = config["max_length"]
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                if len(line) > self.max_length:
                    for i in range(len(line) // self.max_length):
                        input_id, label = self.process_sentence(line[i * self.max_length:(i+1) * self.max_length])
                        self.data.append([torch.LongTensor(input_id), torch.LongTensor(label)])
                else:
                    input_id, label = self.process_sentence(line)
                    self.data.append([torch.LongTensor(input_id), torch.LongTensor(label)])
        return

    def process_sentence(self, line):
        sentence_without_sign = []
        label = []
        for index, char in enumerate(line[:-1]):
            if char in self.schema:  #准备加的标点，在训练数据中不应该存在
                continue
            sentence_without_sign.append(char)
            next_char = line[index + 1]
            if next_char in self.schema:  #下一个字符是标点，计入对应label
                label.append(self.schema[next_char])
            else:
                label.append(0)
        assert len(sentence_without_sign) == len(label)
        encode_sentence = self.encode_sentence(sentence_without_sign)
        label = self.padding(label, -1)
        assert len(encode_sentence) == len(label)
        self.sentences.append("".join(sentence_without_sign))
        return encode_sentence, label

    def encode_sentence(self, text, padding=True):
        input_id = []
        if self.config["vocab_path"] == "vocab.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
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

#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f: #采用bert字表
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)

