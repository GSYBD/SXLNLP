# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import csv

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
        self.index_to_label = {0: '差评', 1: '好评'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        # print('self.vocab',self.vocab,self.config)
        # exit()
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            # csv 文件跳过第一行
            if self.path.lower().endswith('.csv'):
                csvreader = csv.reader(f)
                # 跳过第一行
                next(csvreader)

            for line in f:
                line = self.handel_line(line)
                # print(line)
                # exit()
                # line = json.loads(line)
                tag = line["tag"]
                label = self.label_to_index[tag]
                title = line["title"]
                # print(title,tag,label)
                # exit()
                """
                对于非bert就用自己的词表、索引，处理文本长度
                bert 不一样，所以单独处理，词向量什么的已经有了 
                input id [101, 100, 2208, 2399, 679, 1377, 3619, 100, 8024, 4276, 3326, 
                771, 679, 1377, 3619, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                输入转为索引
                """
                if self.config["model_type"] == "bert":
                    # input_id = self.tokenizer.encode(title, max_length=self.config["max_length"],
                    #                                  pad_to_max_length=True,truncation=True)
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"],
                                                     padding='max_length', truncation=True)
                else:
                    input_id = self.encode_sentence(title)

                # print(input_id)
                # exit()

                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return

    def handel_line(self, line):
        if self.path.lower().endswith('.csv'):
            line = line.split(',', 1)
            tag = self.index_to_label[int(line[0])]
            title = line[1].rstrip()
            line = {"tag": tag, "title": title}
        else:
            line = json.loads(line)
        return line

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
def load_data(data_path, config, shuffle=True):
    """
    dg:将文本转为文字索引
    [tensor([[12],[12]]),tensor([[12],[12]])...]

dl: 初始化一个数据加载器
    :param data_path:
    :param config:
    :param shuffle:
    :return:
    """
    dg = DataGenerator(data_path, config)
    # print(dg.data)
    # exit()
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    # print(dl)
    # exit()
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])
