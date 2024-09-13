'''
数据加载
'''
import json
import random
from collections import defaultdict

import jieba
import torch
from torch.utils.data import DataLoader


class DataGenerator:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        # 加载词典，文本->序列
        self.vocab = load_vocab(config['vocab_path'])
        self.config['vocab_size'] = len(self.vocab)
        # {target1:0,target2:1.....} 标准问序列化
        self.schema = load_schema(config["schema_path"])
        # 由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.train_data_size = config["epoch_data_size"]
        # 用来标识加载的是训练集还是测试集 "train" or "test"
        self.data_type = None
        self.load()

    def load(self):
        # 保存测试集序列化数据
        self.data = []
        # 保存训练集序列化数据 {label1:[input_id1,input_id2,,,],label2:[input_id1,input_id2,,,]...}
        self.knwb = defaultdict(list)
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                # 训练集是dict格式，加载训练集
                if isinstance(line, dict):
                    self.data_type = 'train'
                    target = line['target']
                    # 标准问序列化
                    label = self.schema[target]
                    questions = line['questions']
                    # 将知识库每个问题序列化
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[label].append(input_id)
                else:
                    # 训练集是list类型，加载测试训练集
                    assert isinstance(line, list)
                    self.data_type = 'test'
                    question, target = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label = torch.LongTensor([self.schema[target]])
                    self.data.append([input_id, label])
        return

    def encode_sentence(self, text):
        input_id = []
        # 词->序列
        if self.config['vocab_path'] == 'words.txt':
            words = jieba.lcut(text)
            for word in words:
                input_id.append(self.vocab.get(word, self.vocab['[UNK]']))
        else:
            # 字->序列
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab['[UNK]']))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        # 截取最大长度的文本，多余截断丢弃
        input_id = input_id[:self.config["max_length"]]
        # 文本不够补0
        input_id += [0] * (self.config["max_length"] - len(input_id) + 1)
        return input_id

    def __len__(self):
        if self.data_type == 'train':
            return self.config['epoch_data_size']
        else:
            assert self.data_type == 'test', self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == 'train':
            # 随机采样生成一个训练样本
            return self.random_train_sample()
        else:
            return self.data[index]

    def random_train_sample(self):
        # 所有标准问的序列号列表
        standard_question_index = list(self.knwb.keys())
        p, n = random.sample(standard_question_index, 2)
        if len(self.knwb[p]) < 2:
            return self.random_train_sample()
        else:
            # 从两个标准问的相似问列表中分别采样一个问题
            s1, s2 = random.sample(self.knwb[p], 2)
        s3 = random.choice(self.knwb[n])
        # 得到一个负样本
        return [s1, s2, s3]


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            line = line.strip()
            # 0留给padding位置，所以从1开始
            token_dict[line] = index + 1
    return token_dict


def load_schema(schema_path):
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.loads(f.read())


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config['batch_size'], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("../data/train.json", Config)
    dl = DataLoader(dg, batch_size=Config['batch_size'], shuffle=True)
    print(dg[1])
