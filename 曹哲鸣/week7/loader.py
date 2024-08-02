"""
加载数据
"""

import csv
import random

import torch
from config import config
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader


class DataGnerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.label_index = {0: '差评', 1: '好评'}
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        self.data = []
        with open(self.path, mode='r', newline='', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            data = [row for row in csv_reader]
            data_sample = random.sample(data, 1500)
            for i in data_sample:
                label = int(i["label"])
                sentence = i["review"]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(sentence, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_id = self.encode(sentence)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return


    def encode(self, sentence):
        input_id = []
        for words in sentence:
            input_id.append(self.vocab.get(words, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf-8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1
    return token_dict

def load_data(data_path, config, shuffle=True):
    dg = DataGnerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    dl = load_data("data/文本分类练习.csv", config)
    for index , batch_data in enumerate(dl):
        input_ids, labels = batch_data
        print(labels)
