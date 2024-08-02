# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
import random as random
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
        # self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        # self.config["class_num"] = len(self.index_to_label)
        self.config["class_num"] = 2
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.ori_data = []
        self.train_ori_data = []
        self.valid_ori_data = []
        self.train_data = []
        self.valid_data = []
        self.avg_text_len = 0
        self.negative_sample = 0
        self.positive_sample = 0
        self.train_negative_sample = 0
        self.train_positive_sample = 0
        self.valid_negative_sample = 0
        self.valid_positive_sample = 0
        self.train_data_avg_text_len = 0
        self.valid_data_avg_text_len = 0
    def load(self):
        self.data = []
        # with open(self.path, encoding="utf8") as f:
        #     for line in f:
        #         line = json.loads(line)
        #         tag = line["tag"]
        #         label = self.label_to_index[tag]
        #         title = line["title"]
        #         if self.config["model_type"] == "bert":
        #             input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
        #         else:
        #             input_id = self.encode_sentence(title)
        #         input_id = torch.LongTensor(input_id)
        #         label_index = torch.LongTensor([label])
        #         self.data.append([input_id, label_index])
        if len(self.ori_data) == 0:
            total_text_len = 0
            for index, row in pd.read_csv(self.path, encoding="utf8").iterrows():
                total_text_len += len(row["review"])
                if int(row["label"]) == 0:
                    self.negative_sample += 1
                elif int(row["label"]) == 1:
                    self.positive_sample += 1
                self.ori_data.append([row["review"], row["label"]])
            self.avg_text_len = int(total_text_len / len(self.ori_data))
        random.shuffle(self.ori_data)
        self.train_ori_data = self.ori_data[100:]
        self.valid_ori_data = self.ori_data[:100]
        for review, label in self.ori_data:
            if self.config["model_type"] == "bert":
                input_id = self.tokenizer.encode(review, max_length=self.config["max_length"],
                                                 pad_to_max_length=True)
            else:
                input_id = self.encode_sentence(review)
            input_id = torch.LongTensor(input_id)
            label_index = torch.LongTensor([label])
            self.data.append([input_id, label_index])
        self.valid_data = self.data[:100]
        self.train_data = self.data[100:]
        self.train_negative_sample = 0
        self.train_positive_sample = 0
        self.train_data_avg_text_len = 0
        self.valid_data_avg_text_len = 0
        train_data_total_text_len = 0
        valid_data_total_text_len = 0
        for tod in self.train_ori_data:
            if int(tod[1]) == 0:
                self.train_negative_sample += 1
            elif int(tod[1]) == 1:
                self.train_positive_sample += 1
            train_data_total_text_len += len(tod[0])
        self.train_data_avg_text_len = int(train_data_total_text_len / len(self.train_ori_data))
        for vad in self.valid_ori_data:
            if int(vad[1]) == 0:
                self.valid_negative_sample += 1
            elif int(vad[1]) == 1:
                self.valid_positive_sample += 1
            valid_data_total_text_len += len(vad[0])
        self.valid_data_avg_text_len = int(valid_data_total_text_len / len(self.valid_ori_data))
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
def load_data(dg, config, shuffle=True):
    dg.load()
    train_dl = DataLoader(dg.train_data, batch_size=config["batch_size"], shuffle=shuffle)
    valid_dl = DataLoader(dg.valid_data, batch_size=config["batch_size"], shuffle=shuffle)
    return train_dl, valid_dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("文本分类练习.csv", Config)
    dl = DataLoader(dg, batch_size=64, shuffle=True)
    for data in dl:
        print(data)
    # df = pd.read_csv("文本分类练习.csv", encoding="utf8")
    # for index, row in df.iterrows():
    #         print(row["review"])
