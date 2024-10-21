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
        self.bio_schema = {"B_object":0,
                           "I_object":1,
                           "B_value":2,
                           "I_value":3,
                           "O":4}
        self.attribute_schema = json.load(open(config["schema_path"], encoding="utf8"))
        self.config["bio_count"] = len(self.bio_schema)
        self.config["attribute_count"] = len(self.attribute_schema)
        self.max_length = config["max_length"]
        self.load()
        print("超出设定最大长度的样本数量:%d, 占比:%.3f"%(self.exceed_max_length, self.exceed_max_length/len(self.data)))

    def load(self):
        self.text_data = []
        self.data = []
        self.exceed_max_length = 0
        with open(self.path, encoding="utf8") as f:
            for line in f:
                sample = json.loads(line)
                context = sample["context"]
                object = sample["object"]
                attribute = sample["attribute"]
                value = sample["value"]
                if attribute not in self.attribute_schema:
                    attribute = "UNRELATED"
                self.text_data.append([context, object, attribute, value]) #在测试时使用
                input_id, attribute_label, sentence_label = self.process_sentence(context, object, attribute, value)
                self.data.append([torch.LongTensor(input_id),
                                  torch.LongTensor([attribute_label]),
                                  torch.LongTensor(sentence_label)])
        return

    def process_sentence(self, context, object, attribute, value):
        if len(context) > self.max_length:
            self.exceed_max_length += 1
        object_start = context.index(object)
        value_start = context.index(value)
        input_id = self.encode_sentence(context)
        attribute_label = self.attribute_schema[attribute]
        assert len(context) == len(input_id)
        label = [self.bio_schema["O"]] * len(input_id)
        #标记实体
        label[object_start] = self.bio_schema["B_object"]
        for index in range(object_start + 1, object_start + len(object)):
            label[index] = self.bio_schema["I_object"]
        # 标记属性值
        label[value_start] = self.bio_schema["B_value"]
        for index in range(value_start + 1, value_start + len(value)):
            label[index] = self.bio_schema["I_value"]

        input_id = self.padding(input_id)
        label = self.padding(label, -100)
        return input_id, attribute_label, label

    def encode_sentence(self, text, padding=False):
        input_id = []
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
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)

