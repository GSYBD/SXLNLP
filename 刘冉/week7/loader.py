# -*- coding: utf-8 -*-

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
'''
数据加载
'''

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.label_to_index = {"好评": 1, "差评": 0}
        self.index_to_label = dict((y, x) for x, y in self.label_to_index.items())
        self.config["class_num"] = len(self.index_to_label)
        if "bert" in self.config["model_type"]:
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        self.data = []
        with open(self.path, 'r',encoding="utf8") as f:
            #处理数据集
            data_list = json.load(f)
            for index, line in enumerate(data_list):
                label = int(line[0])
                # tag = self.label_to_index[label]
                title = line[1]
                if "bert" in self.config["model_type"]:
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"],
                                                     pad_to_max_length=True)
                else:
                    input_id = encode_sentence(title, self.config["max_length"], self.vocab)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
                if index > 80:
                    return
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def encode_sentence(text, max_length, vocab):
    input_id = []
    for char in text:
        input_id.append(vocab.get(char, vocab["[UNK]"]))
    input_id = padding(input_id, max_length)
    return input_id
#补齐或者截断输入的内容，使其在同一个batch内运算
def padding(input_id, max_length):
    input_id = input_id[:max_length]
    input_id += [0] * (max_length - len(input_id))
    return input_id
def load_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            word = line.strip()
            vocab[word] = index + 1 #0留给padding位置 所以从1开始
        return vocab

def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

def encode_predict(input_data, config):
    if "bert" in config["model_type"]:
        tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        encode_input = tokenizer.encode(input_data, max_length=config["max_length"],
                                         pad_to_max_length=True)
        return encode_input
    else:
        vocab = load_vocab(config["vocab_path"])
        encode_input = encode_sentence(input_data, config["max_length"], vocab)
        return  encode_input


if __name__ == "__main__":
    from config import Config
    dl = load_data("data/train.json",Config)
