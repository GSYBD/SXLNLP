"""
数据加载
"""
import os
import numpy as np
import json
import re
import os
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


# 获取字表集
def load_vocab(path):
    vocab = {}
    with open(path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            word = line.strip()
            # 0留给padding位置，所以从1开始
            vocab[word] = index + 1
        vocab['unk'] = len(vocab) + 1
    return vocab


class DataGenerator:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.schema = self.load_schema(config["schema_path"])
        self.max_len = config["max_len"]
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        # 中文的语句list
        self.sentence_list = []
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

    def load_data(self):
        self.data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line) > self.max_len:
                    for i in range(len(line) // self.max_len):
                        input_id, label = self.process_sentence(line[i * self.max_len:(i + 1) * self.max_len])
                        self.data.append([torch.LongTensor(input_id), torch.LongTensor(label)])

                else:
                    input_id, label = self.process_sentence(line)
                    self.data.append([torch.LongTensor(input_id), torch.LongTensor(label)])
        return

    # 处理文本 输出为不带target标点的文本 + label(目标是当前char的下一个char是否为target dot)
    def process_sentence(self, sentence):
        sentence_without_target = []
        labels = []
        # sentence[:-1] 因为取了 next_char
        for index, char in enumerate(sentence[:-1]):
            if char in self.schema:
                continue
            # 不是target 标点
            sentence_without_target.append(char)
            next_char = sentence[index + 1]
            if next_char in self.schema:
                labels.append(self.schema[next_char])
            else:
                labels.append(0)
        # 向量化
        input_id = self.sentence_to_index(sentence_without_target)
        labels = self.padding(labels, -1)
        # 保存一下原始的句子
        self.sentence_list.append(' '.join(sentence_without_target))
        return input_id, labels

    # 文本预处理
    # 转化为向量
    def sentence_to_index(self, text):
        input_ids = []
        vocab = self.vocab
        # 使用bert的分词来获取inputId
        if self.config["model_type"] == "bert":
            input_ids = self.tokenizer.encode(text,
                                              padding="max_length",
                                              max_length=self.max_len,
                                              truncation=True)
            return input_ids
        else:
            for char in text:
                input_ids.append(vocab.get(char, vocab['unk']))
        # 填充or裁剪
        input_ids = self.padding(input_ids)
        return input_ids

    # 数据预处理 裁剪or填充
    def padding(self, input_ids, padding_dot=0):
        length = self.config["max_len"]
        padded_input_ids = input_ids
        if len(input_ids) >= length:
            return input_ids[:length]
        else:
            padded_input_ids += [padding_dot] * (length - len(input_ids))
            return padded_input_ids


# 用torch自带的DataLoader类封装数据
def load_data_batch(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    # DataLoader 类封装数据 dg除了data 还包含其他信息（后面需要使用）
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == '__main__':
    from config import Config

    dg = DataGenerator(Config["train_data_path"], Config)
    print(len(dg))
    print(dg[0])
