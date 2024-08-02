# -*- coding: utf-8 -*-
"""
author: Chris Hu
date: 2024/8/1
desc:
sample
"""

import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


class DataGenerator(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        # self.index_to_label = config["index_to_label"]
        # self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.label_to_index = config["label_to_index"]
        self.config["class_num"] = len(self.label_to_index)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(self.config["pretrain_model_path"])
        self.vocab = load_vocab(self.config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.data = []
        self.load()

    def load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                tag = line["tag"]
                label = self.label_to_index[tag]

                # todo: need to change below code
                title = line["title"]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"],
                                                     pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append((input_id, label_index))

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
    with open(vocab_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            # 0留给padding位置，所以从1开始
            token_dict[token] = index + 1
    return token_dict


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    data_gen = DataGenerator(data_path, config)
    dl = DataLoader(data_gen, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == '__main__':
    from config import Config

    dg = DataGenerator(r'./data/valid_tag_news.json', Config)
    print(dg[1])
