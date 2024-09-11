# coding utf-8

import json
import torch
import random
import jieba
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

"""
加载数据
"""


class DataGenerator:
    def __init__(self, data_path, config, data_type):
        self.path = data_path
        self.config = config
        self.max_length = config["max_length"]
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        # 由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.train_data_size = config["epoch_data_size"]
        # 用来标识加载的是训练集还是测试集 "train" or "test" 默认为train
        self.data_type = data_type
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                if self.data_type == "train":
                    assert isinstance(line, dict)
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = encode_sentence(question,self.max_length, self.vocab)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                elif self.data_type == "test":
                    assert isinstance(line, list)
                    question, label = line
                    input_id = encode_sentence(question, self.max_length, self.vocab)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])
        return

    def __len__(self):
        if self.data_type == "train":
            return self.train_data_size
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample()  #随机生成一个训练样本
        else:
            return self.data[index]

    # 取 a,p,n  要求a p 同一类型 a n 不同类型
    def random_train_sample(self):
        question_index = list(self.knwb.keys())
        # 随机取两个类型
        r_key1, r_key2 = random.sample(question_index, 2)
        if len(self.knwb[r_key1]) < 2:
            # 如果r_key 内的问题数量不足两个则重新选择
            return self.random_train_sample()
        else:
            # 从第一个类型随机取两个样本
            s1, s2 = random.sample(self.knwb[r_key1], 2)
            # 从第二个类型随机取一个样本
            s3 = random.choice(self.knwb[r_key2])
            return [s1, s2, s3]


def encode_sentence(text, max_length, vocab):
    input_id = []
    # 如果vocab是 chars表 则直接用chars表加载
    for char in text:
        input_id.append(vocab.get(char, vocab["[UNK]"]))
    input_id = padding(input_id, max_length)
    return input_id


def padding(input_id, max_length):
    input_id = input_id[:max_length]
    input_id += [0] * (max_length - len(input_id))
    return input_id

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            # 0留给padding位置，所以从1开始
            token_dict[token] = index + 1
    return token_dict


def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())


def encode_predict(config, text):
    vocab = load_vocab(config["vocab_path"])
    encode_input = encode_sentence(text, config["max_length"], vocab)
    return encode_input


# 用来标识加载的是训练集还是测试集 "train" or "test" 默认为train
def load_data(data_path, config, data_type="train", shuffle=True):
    dg = DataGenerator(data_path, config, data_type=data_type)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
