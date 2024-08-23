import json
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from config import Config

"""

    数据加载

"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        # 生成字表
        # self.vocab = load_vocab(config['vocab_path'])
        self.vocab = BertTokenizer.from_pretrained(config['bert_path'])
        # 字表长度, 不包括padding
        self.config['vocab_size'] = len(self.vocab)
        # label字典, {str: 0, ..., str: int}
        self.schema = self.load_schema()
        self.config['class_num'] = len(self.schema)
        self.data = []
        self.sentences = []  # evaluate.py
        self.load()

    def load_schema(self):
        with open(self.config['schema_path'], encoding='utf-8') as f:
            return json.load(f)

    def load(self):
        with open(self.data_path, encoding='utf-8') as f:
            # 每个字加标签为一行, 空行分割每句话
            segments = f.read().split('\n\n')
            for segment in segments:
                sentence = []
                labels = [8]  # cls_token
                # 按每个label后的换行划分
                for line in segment.split('\n'):
                    if len(line.split()) < 2:
                        continue
                    # 分割字和标签
                    char, label = line.split()
                    # [char, char, ..., char]
                    sentence.append(char)
                    # [int, int, ..., int]
                    labels.append(self.schema[label])
                self.sentences.append(''.join(sentence))
                input_ids = self.encode_sentence(sentence)
                # -1为无用类, 对齐input_id中的补零
                labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])

    def encode_sentence(self, text, padding=True):
        return self.vocab.encode(text,
                                 padding="max_length",
                                 max_length=self.config["max_length"],
                                 truncation=True)

    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config['max_length']]
        input_id += [pad_token] * (self.config['max_length'] - len(input_id))
        return input_id

    def __len__(self):
        # 句子数
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            token = line.strip()
            token_dict[token] = idx + 1
    return token_dict


def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config['batch_size'], shuffle=shuffle)
    return dl


if __name__ == '__main__':
    data_generator = DataGenerator('ner_data/train', Config)
    print(data_generator[0])
    print(len(data_generator.schema))
