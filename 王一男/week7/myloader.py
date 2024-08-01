# -*- coding: utf-8 -*-

import json
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '负面', 1: '正面'}  # 二分类
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])  # 可选，若不需要可移除
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        self.data = []
        df = pd.read_csv(self.path, encoding="utf8")  # 使用pandas读取CSV文件
        for index, row in df.iterrows():
            label = row["label"]
            review = row["review"]
            if self.config["model_type"] == "bert":
                input_id = self.tokenizer.encode(review, max_length=self.config["max_length"], padding="max_length")
            else:
                input_id = self.encode_sentence(review)
            input_id = torch.LongTensor(input_id)  # 对每一句文本编码，逐行添加进data形成batches
            label_index = torch.LongTensor([label])
            self.data.append((input_id, label_index))

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

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
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


def split_data(input_csv, train_csv, test_csv, test_size=0.2, random_state=42):
    df = pd.read_csv(input_csv)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)


if __name__ == "__main__":
    from config import Config

    # 数据拆分
    input_csv_path = Config["data_path"]  # 输入CSV文件路径
    train_csv_path = os.path.dirname(Config["data_path"])+"\\train_reviews.csv"  # 训练集保存路径
    test_csv_path = os.path.dirname(Config["data_path"])+"\\test_reviews.csv"  # 测试集保存路径
    split_data(input_csv_path, train_csv_path, test_csv_path)


# # 使用pandas写入CSV
# results_df = pd.DataFrame({
#     'Training Time (s)': [training_time],
#     'Accuracy': [accuracy],
#     'Model Structure': [model_structure]
# })
#
# # 写入新CSV文件
# results_df.to_csv('results.csv', index=False)

# 或者使用csv模块（更基础的方式）
# import csv
#
# with open('results.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Training Time (s)', 'Accuracy', 'Model Structure'])
#     writer.writerow([training_time, accuracy, model_structure])

