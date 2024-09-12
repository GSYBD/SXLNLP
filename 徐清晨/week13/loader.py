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
    """
    每句话变成这种形式，对应的id，以及对应的标签
    [ 265, 3778,   27,  185,  868, 1803, 1320, 1163, 2795,  525,  597,  232,
         489, 2609, 2769, 2025,  454,  969, 3004, 3881, 2769, 1192,  552, 2344,
        1508, 1418, 3574,  727,  165, 1117,  145,。。。。
    [8, 8, 8, 1, 5, 5, 5, 8, 3, 7, 0, 4, 8, 8, 8, 8, 8, 8, 8
    , 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, -1, -1,
    """

    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")

            for segment in segments:
                sentenece = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)
                    labels.append(self.schema[label])

                self.sentences.append("".join(sentenece))


                encoder = self.tokenizer.encode_plus(sentenece,max_length=self.config['max_length'],
                                                  padding='max_length',truncation=True)
# [101, 800, 6432, 131, 704, 1744, 3124, 2424, 2190, 4680, 1184, 1298, 762, 1139, 4385, 4638, 3417, 1092, 1906, 4993, 6612, 4638, 2229, 1232, 3918, 2697, 2569, 5991, 1469, 679, 2128, 511, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 50
                # [101, 800, 6432, 131, 704, 1744, 3124, 2424, 2190, 102]
                input_ids = encoder['input_ids']
                attention_mask = encoder["attention_mask"]
                # print(labels)
                labels = self.padding_label(labels,-1)
                # print(input_ids,len(input_ids))
                # print(labels)
                # print(input_ids.index(102))
                # exit()
                self.data.append([torch.LongTensor(input_ids),  torch.LongTensor(attention_mask),
                                  torch.LongTensor(labels)])

        return

    def encode_sentence(self, text, padding=True):
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

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def padding_label(self, input_id, pad_token=0):

        input_id = input_id[:self.config["max_length"]-2]
        input_id.insert(0, 8)
        input_id.append(8)
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        # input_id[input_ids.index(102)] = -1
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)


# 加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("../ner_data/train.txt", Config)
