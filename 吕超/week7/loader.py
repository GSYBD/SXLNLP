# -*- coding: utf-8 -*-
import csv
import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        # self.index_to_label = {0: '家居', 1: '房产', 2: '股票', 3: '社会', 4: '文化',
        #                        5: '国际', 6: '教育', 7: '军事', 8: '彩票', 9: '旅游',
        #                        10: '体育', 11: '科技', 12: '汽车', 13: '健康',
        #                        14: '娱乐', 15: '财经', 16: '时尚', 17: '游戏'}
        self.index_to_label = {0: '0', 1: '1'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)   #  lyu: 起到什么作用？
        self.load()


    # def load(self):
    #     self.data = []
    #     with open(self.path, encoding="utf8") as f:
    #         for line in f:
    #             line = json.loads(line)
    #             tag = line["tag"]  # lyu: 获取tag对应的值, 进而获取label_to_index中的索引值, 作为标签最后进行输出
    #             label = self.label_to_index[tag]
    #             title = line["title"]
    #             if self.config["model_type"] == "bert":  # lyu: 如果是bert模型，则使用tokenizer对标题进行编码，否则使用自定义的编码方式
    #                 input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
    #             else:
    #                 input_id = self.encode_sentence(title)
    #             input_id = torch.LongTensor(input_id)
    #             label_index = torch.LongTensor([label])
    #             self.data.append([input_id, label_index])
    #     return

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            csv_reader = csv.reader(f)
            # next(csv_reader, None)  # 读取并丢弃文件的第一行
            for line in csv_reader:
                label_str = line[0]  # 第一个字段作为label
                label = self.label_to_index[label_str]  # lyu: 获取label对应的值, 进而获取label_to_index中的索引值, 注意去掉字段名称避免报找不到key
                review = line[1]  # 第二个字段作为review
                if self.config["model_type"] == "bert":  # lyu: 如果是bert模型，则使用tokenizer对标题进行编码，否则使用自定义的编码方式
                    input_id = self.tokenizer.encode(review, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(review)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])  # lyu: input_ids(128*30), labels(128*1)是batch_data中的前两个元素
        return

    def encode_sentence(self, text):
        input_id = []
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


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    # dg = DataGenerator("../data/tag_news.json", Config)
    dg = DataGenerator("../data/my_val_data.csv", Config)
    print(dg[0])
    print(len(dg))




