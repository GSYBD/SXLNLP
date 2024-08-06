# -*- coding: utf-8 -*-

import json
import random
import re
import os
import torch
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '0', 1: '1'} #0-差评 1-好评
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []
        with open(self.path,encoding="utf8") as f:
            csv_reader = csv.reader(f)
            max_len = 0
            for row in csv_reader:
                if csv_reader.line_num == 1 :
                    continue
                if len(row)<2:
                    continue
                label = row[0]
                text = row[1]
                if label not in self.label_to_index:
                    continue
                max_len=max(len(text),max_len)
                # print("%s--->%s"%(text,label))

                if self.config["model_type"] == "bert":
                    input = self.tokenizer.encode(text, max_length=self.config["max_length"], padding='max_length')
                else:
                    input = self.encode_sentence(text)
                label = self.label_to_index[label]
                self.data.append([torch.LongTensor(input),torch.LongTensor([label])])
            print('样本数据最大长度(配置到配置文件中):',max_len)
        return


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
        # return

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

def build_eval_data():
    #随机抽取一部分数据当测试数据,验证正确率
    eval_x = random.sample(range(50, 8000), 200)
    eval_data = []

    with open("D:/ai/week7 文本分类问题/文本分类练习.csv",encoding="utf8") as f:
        csv_r = csv.reader(f)
        for row in csv_r:
            if csv_r.line_num == 1:
                continue
            if csv_r.line_num in eval_x:
                eval_data.append(row)

    random.shuffle(eval_data)
    random.shuffle(eval_data)
    random.shuffle(eval_data)
    random.shuffle(eval_data)
    random.shuffle(eval_data)
    random.shuffle(eval_data)
    random.shuffle(eval_data)
    random.shuffle(eval_data)
    random.shuffle(eval_data)
    # 写入文件
    with open("D:/ai/week7 文本分类问题/文本分类练习_测试正确率.csv","a",encoding="utf8",newline="") as f:
        csv_w = csv.writer(f)
        for x in eval_data:
            csv_w.writerow(x)


if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("D:/ai/week7 文本分类问题/文本分类练习.csv", Config)
    dl = DataLoader(dg, batch_size=Config["batch_size"], shuffle=True)
    for index,data in enumerate(dl):
        print(index)
        # print(data)
    # build_eval_data()

