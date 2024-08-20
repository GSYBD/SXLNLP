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
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        # main(训练）或evaluate（预测）传过来的路径
        self.path = data_path
        # 使用bert自己的词表
        self.tokenizer = load_vocab(config["bert_path"])
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        # 准备了每句话中每个字符（或词）的编码（input_ids）以及它们对应的真实标签（用于计算损失），所以因为cls的添加，所以label在最前面需要补有一个对应的
        # 打开训练数据集train
        with open(self.path, encoding="utf8") as f:
            # 根据双换行符("\n\n")分割成多个段落
            segments = f.read().split("\n\n")
            for segment in segments:
                sentenece = []
                # 如果使用bert自带的tokenizer，bert会在编码后的第一个位置加cls_token
                # 因为bert模型在处理文本时，会首先添加一个CLS标记，所以标签列表也要添加一个对应的标签8
                # 输入给BERT模型的序列就会由[CLS] token加上句子本身对应的编码序列再加上[SEP] token组成
                labels = [8]
                # 遍历每一行，根据空格分割成字符和标签
                for line in segment.split("\n"):
                    # 如果是空行，则跳过
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)
                    # 将标签转换为schema.json中定义的数字
                    labels.append(self.schema[label])
                sentence = "".join(sentenece)
                # 将句子添加到句子列表中
                self.sentences.append(sentence)
                # 对句子字符列表进行编码，得到输入ID列表 
                input_ids = self.encode_sentence(sentenece)
                # 多余的label填充为-1
                labels = self.padding(labels, -1)
                # print(self.decode(sentence, labels))
                # input()
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def encode_sentence(self, text, padding=True):
        return self.tokenizer.encode(text, 
                                     padding="max_length",
                                     max_length=self.config["max_length"],
                                     truncation=True)

    def decode(self, sentence, labels):
        sentence = "$" + sentence
        labels = "".join([str(x) for x in labels[:len(sentence)+2]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            print("location", s,e)
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            print("org", s,e)
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            print("per", s,e)
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            print("time", s,e)
            results["TIME"].append(sentence[s:e])
        return results
    

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

def load_vocab(vocab_path):
    return BertTokenizer.from_pretrained(vocab_path)


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("ner_data/train", Config)
    dl = DataLoader(dg, batch_size=32)  
    for x,y in dl:
        print(x.shape, y.shape)
        print(x[1], y[1])
        input()