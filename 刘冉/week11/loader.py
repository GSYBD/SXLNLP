# coding: utf-8

import json
import torch
import os
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

'''
数据加载
'''
class DataGenerator:
    def __init__(self, config, data_path):
        self.config = config
        self.path = data_path
        self.max_length = config["max_length"]
        self.bertTokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                input_seq, label_seq, mask = self.line_to_sentence(line, self.max_length)
                input_seqs = torch.LongTensor(input_seq)
                label_seqs = torch.LongTensor(label_seq)
                self.data.append([input_seqs, label_seqs, mask])
    #从数据中取出没有标点的sentences和lables 最大长度为max_length
    def line_to_sentence(self, line, max_length):
        jsonDict = json.loads(line)
        input = []
        label = []
        title = jsonDict["title"]
        content = jsonDict["content"]
        title_length = len(title)+1
        content_length = len(content)
        top_left_mask = torch.ones(title_length, title_length)
        top_right_mask = torch.zeros(title_length, content_length)
        top_mask = torch.cat([top_left_mask, top_right_mask], dim=1)
        bottom_left_mask = torch.ones(content_length, title_length)
        bottom_right__mask = torch.tril(torch.ones(content_length, content_length))
        bottom_mask = torch.cat([bottom_left_mask, bottom_right__mask], dim=1)
        mask = torch.cat([top_mask, bottom_mask], dim=0)
        mask = torch.nn.functional.pad(mask, (0, max_length - mask.shape[1], 0, max_length - mask.shape[0]), "constant", 0)
        for index, char in enumerate(title):
            input.append(char)
            label.append('[PAD]')
        input.append('[SEP]')
        for index, char in enumerate(content):
            input.append(char)
            label.append(char)
        label.append('[EOS]')
        input_seq = self.bertTokenizer.encode(input, max_length=max_length, add_special_tokens=False, truncation=True, padding="max_length")
        label_seq = self.bertTokenizer.encode(label, max_length=max_length, add_special_tokens=False, truncation=True, padding="max_length")
        return input_seq, label_seq, mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_data(config, data_path, shuffle=True):
    dg = DataGenerator(config, data_path)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl