# -*- coding: utf-8 -*-

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from config import Config
import pandas as pd
from sklearn.model_selection import train_test_split

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data, config):
        self.config = config
        self.data = data
        self.index_to_label = {0: "负面评价", 1: "正面评价"}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                tag = line["tag"]
                label = self.label_to_index[tag]
                title = line["title"]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"],
                                                     pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return

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
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


def load_data(data_path, config, shuffle=True):
    df = pd.read_csv(data_path, encoding="utf8")
    data = [(int(row.iloc[0]), row.iloc[1]) for _, row in df.iterrows()]
    train_data, test_data = train_test_split(data, test_size=config["test_size"], random_state=config["random_state"], shuffle=True)

    train_dataset = DataGenerator(train_data, config)
    test_dataset = DataGenerator(test_data, config)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    return train_loader, test_loader

if __name__ == "__main__":

    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])
