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
    def __init__(self, data_path, config, tokenizer):
        self.config = config
        self.path = data_path
        self.tokenizer = tokenizer
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentence))
                inputs = self.tokenizer("".join(sentence), max_length=self.config["max_length"], padding="max_length",truncation=True, return_tensors="pt")
                labels = self.padding(labels, -1)
                inputs['labels'] = torch.LongTensor(labels).unsqueeze(0)
                self.data.append(inputs)
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

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, labels, pad_token=-1):
        labels = labels[:self.config["max_length"]]
        labels += [pad_token] * (self.config["max_length"] - len(labels))
        return labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

#加载字表或词表
# def load_vocab(vocab_path):
#     token_dict = {}
#     with open(vocab_path, encoding="utf8") as f:
#         for index, line in enumerate(f):
#             token = line.strip()
#             token_dict[token] = index + 1  #0留给padding位置，所以从1开始
#     return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
    dg = DataGenerator(data_path, config, tokenizer)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)

