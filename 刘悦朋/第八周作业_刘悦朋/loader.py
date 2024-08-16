import json
import random
import torch
from collections import defaultdict
from config import Config
from torch.utils.data import DataLoader

"""

    数据加载

"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.data_type = None  # 用来标识加载的是训练集还是测试机, 'train' or 'test'
        self.config = config
        self.knwb = defaultdict(list)
        self.schema = load_schema(config['schema_path'])  # label_dict
        self.vocab = load_vocab(config['vocab_path'])  # token_dict
        self.config['vocab_size'] = len(self.vocab)
        self.data = []
        self.train_data_size = config['epoch_data_size']
        self.load()

    def load(self):
        with open(self.data_path, encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                # 加载训练集
                if isinstance(line, dict):
                    self.data_type = 'train'
                    questions = line['questions']
                    label = line['target']
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                # 加载测试集
                else:
                    self.data_type = 'test'
                    assert isinstance(line, list)
                    question, label = line  # [question: str, label: str]
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab['[UNK]']))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config['max_length']]
        input_id += [0] * (self.config['max_length'] - len(input_id))
        return input_id

    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        # 随机正样本
        if random.random() <= self.config['positive_sample_rate']:
            p = random.choice(standard_question_index)
            if len(self.knwb[p]) < 2:
                return self.random_train_sample()
            else:
                s1, s2 = random.sample(self.knwb[p], 2)
                return [s1, s2, torch.LongTensor([1])]
        # 随机负样本
        else:
            p, n = random.sample(standard_question_index, 2)
            s1 = random.choice(self.knwb[p])
            s2 = random.choice(self.knwb[n])
            return [s1, s2, torch.LongTensor([-1])]

    def triplet_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        p, n = random.sample(standard_question_index, 2)
        s1 = random.choice(self.knwb[p])
        s2 = random.choice(self.knwb[p])
        s3 = random.choice(self.knwb[n])
        return [s1, s2, s3]

    def __len__(self):
        if self.data_type == 'train':
            return self.config['epoch_data_size']
        else:
            return len(self.data)

    def __getitem__(self, item):
        if self.data_type == 'train':
            return self.triplet_train_sample()  # 随机生成一个训练样本
        else:
            return self.data[item]


# 加载schema, 即label_dict
def load_schema(schema_path):
    """
    return -> dict：{str : int, str: int, ..., str: int}
    """
    with open(schema_path, encoding='utf-8') as f:
        return json.loads(f.read())


# 加载字表
def load_vocab(vocab_path):
    """
    return -> token_dict：{str: int, str:, int, ..., str: int}
    """
    token_dict = {}
    with open(vocab_path, encoding='utf-8') as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置, 所以从1开始
    return token_dict


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config['batch_size'], shuffle=shuffle)
    return dl


if __name__ == '__main__':
    dg = DataGenerator('data/data.json', Config)
    dl = load_data('data/data.json', Config)
    if dg.data_type == 'train':
        samples = dg[0]
        for sample in samples:
            sen = []
            for char in list(sample):
                if int(char) != 0:
                    sen.append(list(dg.vocab.keys())[list(dg.vocab.values()).index(int(char))])
            print(''.join(sen))
