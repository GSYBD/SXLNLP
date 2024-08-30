# -*- coding: utf-8 -*-

import json
import re
import os
from collections import defaultdict

import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


def build_sample(title, content, max_length, tokenizer):
    input_id = title + "[SEP]" + content
    target_id = content

    x = tokenizer.encode(input_id, add_special_tokens=False, max_length=max_length,
                         truncation=True, padding="max_length")
    y = tokenizer.encode(target_id, add_special_tokens=False, max_length=(max_length - len(title)),
                         truncation=True, padding="max_length")
    y = [0]*len(title) + y
    mask = np.zeros((max_length, max_length), dtype=int)
    real_len = min(len(input_id), max_length)
    mask[0:real_len, 0:len(title) + 1] = 1
    for i in range(len(title) + 1, real_len):
        mask[i:real_len, i:i + 1] = 1
    return x, y, mask


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = load_vocab(config["vocab_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                content = line["content"]
                title = line["title"]
                x, y, mask = build_sample(title, content, self.config["max_length"], self.tokenizer)
                self.data.append([torch.LongTensor(x), torch.LongTensor(y), torch.LongTensor(mask)])
        return



    # def encode_sentence(self, text, padding=True):
    #     input_id = []
    #     if self.config["vocab_path"] == "words.txt":
    #         for word in jieba.cut(text):
    #             input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
    #     else:
    #         for char in text:
    #             input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
    #     if padding:
    #         input_id = self.padding(input_id)
    #     return input_id



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]



# 加载字表或词表
def load_vocab(vocab_path):
    tokenizer = BertTokenizer(vocab_path)
    return tokenizer


# #加载字表或词表
# def load_vocab(vocab_path):
#     token_dict = {}
#     with open(vocab_path, encoding="utf8") as f:
#         for index, line in enumerate(f):
#             token = line.strip()
#             token_dict[token] = index + 1  #0留给padding位置，所以从1开始
#     return token_dict

# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dl = load_data(r"D:\资料\week10 文本生成问题\bert_sft\sample_data.json", Config)
    for x, y, mask in dl:
        print(x.shape, y.shape, mask.shape)
        print(x[0], y[0], mask[0])
        input()
