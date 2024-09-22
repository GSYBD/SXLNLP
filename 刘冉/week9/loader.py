# coding: utf-8

import json
import torch
import os
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

'''
数据加载
'''


def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.load(f)


class DataGenerator:
    def __init__(self, config, data_path):
        self.config = config
        self.path = data_path
        self.max_length = config["max_length"]
        self.schema = load_schema(config["schema_path"])
        config["class_num"] = len(self.schema)
        self.padding = config["padding"]
        self.bertTokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"], pad_token_id=self.padding)
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                sentences, labels = self.line_to_sentence(line, self.max_length)
                encode_inputs = self.bertTokenizer.encode(sentences, max_length=self.max_length, padding="max_length")
                encode_inputs = torch.LongTensor(encode_inputs)
                # encode_sen 多了[SEP]和[CLS]两个占位符，所以labels也需要补位
                labels.insert(0, self.padding)
                labels.append(self.padding)
                labels = torch.LongTensor(labels)
                self.data.append([encode_inputs, labels])

    #从数据中取出没有标点的sentences和lables 最大长度为max_length
    def line_to_sentence(self, line, max_length):
        sentences = []
        labels = []
        for index, char in enumerate(line):
            if char in self.schema:
                continue
            sentences.append(char)
            if index + 1 == len(line):
                labels.append(0)
            else:
                next_char = line[index + 1]
                if next_char in self.schema:
                    labels.append(self.schema[next_char])
                else:
                    labels.append(0)
            if len(sentences) >= max_length:
                break
        #把字数补到max_length
        sentences += [self.padding] * (self.config["max_length"] - len(sentences))
        labels += [self.padding] * (self.config["max_length"] - len(labels))
        return sentences, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_data(config, data_path, shuffle=True):
    dg = DataGenerator(config, data_path)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl