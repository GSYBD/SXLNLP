
"""
数据加载
"""
import json
from collections import defaultdict
import random

import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
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


# 数据预处理 裁剪or填充
def padding(input_ids, length):
    if len(input_ids) >= length:
        return input_ids[:length]
    else:
        padded_input_ids = input_ids + [0] * (length - len(input_ids))
        return padded_input_ids


# 文本预处理
# 转化为向量
def sentence_to_index(text, length, vocab):
    input_ids = []
    for char in text:
        input_ids.append(vocab.get(char, vocab['unk']))
    # 填充or裁剪
    input_ids = padding(input_ids, length)
    return input_ids


class DataGenerator:
    def __init__(self, data_path, config):
        # 加载json数据
        self.load_know_base(config["train_data_path"])
        # 加载schema 相当于答案集
        self.schema = self.load_schema(config["schema_path"])
        self.data_path = data_path
        self.config = config
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.train_flag = None
        self.load_data()

    def __len__(self):
        if self.train_flag:
            return self.config["simple_size"]
        else:
            return len(self.data)

    # 这里需要返回随机的样本
    def __getitem__(self, idx):
        if self.train_flag:
            # return self.random_train_sample()  # 随机生成一个训练样本
            # triplet loss:
            return self.random_train_sample_for_triplet_loss()
        else:
            return self.data[idx]

    # 针对获取的文本 load_know_base = ｛target : [questions]｝ 做处理
    # 传入两个样本 正样本为相同target数据 负样本为不同target数据
    # 训练集和验证集不一致
    def load_data(self):
        self.train_flag = self.config["train_flag"]
        dataset_x = []
        dataset_y = []
        self.knwb = defaultdict(list)
        if self.train_flag:
            for target, questions in self.target_to_questions.items():
                for question in questions:
                    input_id = sentence_to_index(question, self.config["max_len"], self.vocab)
                    input_id = torch.LongTensor(input_id)
                    # self.schema[target] 下标 把每个question转化为向量append放入一个target下
                    self.knwb[self.schema[target]].append(input_id)
        else:
            with open(self.data_path, encoding="utf8") as f:
                for line in f:
                    line = json.loads(line)
                    assert isinstance(line, list)
                    question, target = line
                    input_id = sentence_to_index(question, self.config["max_len"], self.vocab)
                    # input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[target]])
                    # self.data.append([input_id, label_index])
                    dataset_x.append(input_id)
                    dataset_y.append(label_index)
                self.data = Data.TensorDataset(torch.tensor(dataset_x), torch.tensor(dataset_y))
        return

    # 加载知识库
    def load_know_base(self, know_base_path):
        self.target_to_questions = {}
        with open(know_base_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                content = json.loads(line)
                questions = content["questions"]
                target = content["target"]
                self.target_to_questions[target] = questions
        return

    # 加载schema 相当于答案集
    def load_schema(self, param):
        with open(param, encoding="utf8") as f:
            return json.loads(f.read())

    # 训练集随机生成一个样本
    # 依照一定概率生成负样本或正样本
    # 负样本从随机两个不同的标准问题中各随机选取一个
    # 正样本从随机一个标准问题中随机选取两个
    def random_train_sample(self):
        target = random.choice(list(self.knwb.keys()))
        # 随机正样本：
        # 随机正样本
        if random.random() <= self.config["positive_sample_rate"]:
            if len(self.knwb[target]) <= 1:
                return self.random_train_sample()
            else:
                question1 = random.choice(self.knwb[target])
                question2 = random.choice(self.knwb[target])
                # 一组
                # dataset_x.append([question1, question2])
                # # 二分类任务 同一组的question target = 1
                # dataset_y.append([1])
                return [question1, question2, torch.LongTensor([1])]
        else:
            # 随机负样本：
            p, n = random.sample(list(self.knwb.keys()), 2)
            question1 = random.choice(self.knwb[p])
            question2 = random.choice(self.knwb[n])
            # dataset_x.append([question1, question2])
            # dataset_y.append([-1])
            return [question1, question2, torch.LongTensor([-1])]

    # triplet_loss随机生成3个样本 锚样本A, 正样本P, 负样本N
    def random_train_sample_for_triplet_loss(self):
        target = random.choice(list(self.knwb.keys()))
        # question1锚样本 question2为同一个target下的正样本 question3 为其他target下样本
        question1 = random.choice(self.knwb[target])
        question2 = random.choice(self.knwb[target])
        question3 = random.choice(self.knwb[random.choice(list(self.knwb.keys()))])
        return [question1, question2, question3]


# 用torch自带的DataLoader类封装数据
def load_data_batch(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    if config["train_flag"]:
        dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    else:
        dl = DataLoader(dg.data, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == '__main__':
    from config import Config

    Config["train_flag"] = True
    # dg = DataGenerator(Config["train_data_path"], Config)
    dataset = load_data_batch(Config["train_data_path"], Config)
    # print(len(dg))
    # print(dg[0])
    for index, dataset in enumerate(dataset):
        input_id1, input_id2, input_id3 = dataset
        print(input_id1)
        print(input_id2)
        print(input_id3)
