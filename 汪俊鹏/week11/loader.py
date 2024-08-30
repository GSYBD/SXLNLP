# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config, logger):
        self.config = config
        self.logger = logger
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.config["middle_idx"] = self.vocab["[CLS]"]
        self.config["end_idx"] = self.vocab["[SEP]"]
        self.load()
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])


    def load(self):
        self.data = []
        self.matrices = []
        with open(self.path, encoding="utf8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                self.prepare_data(title, content)
        return

    #文本到对应的index
    #头尾分别加入[cls]和[sep]
    def encode_sentence(self, title,contex, max_length,with_mid_idx=True,with_end_idx=True):
        input_id = []
        if with_mid_idx:
            for char in title:
                input_id.append(self.vocab.get(char,self.vocab['[UNK]']))
            input_id.append(self.vocab['[CLS]'])
            ask_len=len(input_id)
            ask_mask=torch.ones(ask_len,ask_len)
            for char in contex:
                input_id.append(self.vocab.get(char,self.vocab['[UNK]']))
            que_len=len(input_id)-ask_len
            que_mask=torch.zeros(ask_len,que_len)
            Look_Ahead_mask = torch.tril(torch.ones(ask_len, que_len))
            mask_end_up = torch.cat((ask_mask, que_mask), dim=1)
            mask_end_down = torch.cat((ask_mask, Look_Ahead_mask), dim=1)
            mask_end = torch.cat((mask_end_up, mask_end_down), dim=0)
            return  input_id,mask_end
        if with_end_idx:
            for char in contex:
                input_id.append(self.vocab.get(char, self.vocab['[UNK]']))
            input_id.append(self.vocab['[SEP]'])

        input_id = self.padding(input_id, max_length)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, length):
        input_id = input_id[:length]
        input_id += [self.vocab["[PAD]"]] * (length - len(input_id))
        return input_id


    #输入输出转化成序列
    def prepare_data(self, title, content):
        input_seq ,mask= self.encode_sentence(title,content, self.config["max_length"], True, False) #输入序列
        output_seq= self.encode_sentence(title,content, self.config["max_length"], False, True) #输出序列
        pad_length=len(input_seq)-len(output_seq)
        padding_value = [-99]
        output_seq=pad_length*padding_value+output_seq
        self.data.append([torch.LongTensor(input_seq),
                          torch.LongTensor(output_seq)])
        self.matrices.append(mask)
        return


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, logger, shuffle=True):
    dg = DataGenerator(data_path, config, logger)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

def load_mask(data_path, config, logger, shuffle=True):
    dg = DataGenerator(data_path, config, logger)
    mask=torch.stack(dg.matrices)
    return mask

if __name__ == "__main__":
    from config import Config
    dl = load_data(Config["train_data_path"], Config, 1)
    print(dl[1])

