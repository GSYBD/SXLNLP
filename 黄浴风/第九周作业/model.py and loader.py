"""
model.py
"""

# -*- coding: utf-8 -*-
"""
建立网络模型结构
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from TorchCRF import CRF
from torch.optim import Adam, SGD

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"])  # 加载预训练的BERT模型
        self.class_num = config["class_num"]
        self.dropout = nn.Dropout(0.1)
        self.classify = nn.Linear(self.bert.config.hidden_size, self.class_num)
        self.crf_layer = CRF(self.class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # 交叉熵损失函数

    def forward(self, x, target=None):
        attention_mask = (x != 0).float()  # 生成attention mask，避免对padding部分进行编码
        outputs = self.bert(x, attention_mask=attention_mask)
        sequence_output = outputs[0]  # 获取最后一层的隐藏状态
        sequence_output = self.dropout(sequence_output)  # 添加dropout
        predict = self.classify(sequence_output)  # 通过全连接层进行分类

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                return self.loss(predict.view(-1, self.class_num), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict



def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)



"""
loader.py
"""
# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader


"""
数据加载
"""



from transformers import BertTokenizer

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])  # 初始化BERT Tokenizer
        self.sentences = []  # 初始化 self.sentences 为空列表
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentence))  # 将句子加入到 self.sentences 列表中
                input_ids, label_ids = self.encode_sentence(sentence, labels)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(label_ids)])
        return

    def encode_sentence(self, sentence, labels):
        tokens = []
        label_ids = []
        for i, word in enumerate(sentence):
            tokenized_word = self.tokenizer.tokenize(word)
            tokens.extend(tokenized_word)
            label_ids.extend([labels[i]] + [-1] * (len(tokenized_word) - 1))  # 对子词标记为-1

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = self.padding(input_ids)
        label_ids = self.padding(label_ids, -1)
        return input_ids, label_ids

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

#加载字表或词表
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



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)

