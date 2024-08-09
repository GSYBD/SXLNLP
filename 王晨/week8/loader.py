# -*- coding: utf-8 -*-

# loader.py

import json
import random
import jieba
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from config import Config

class DataGenerator(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = self.load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = self.load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"]  # Number of samples to use for training
        self.data_type = None
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample()  # Randomly generate a training sample
        else:
            return self.data[index]

    def random_train_sample(self):
        # Get all standard question categories
        standard_question_index = list(self.knwb.keys())
        # Randomly choose a category as anchor
        anchor_class = random.choice(standard_question_index)
        if len(self.knwb[anchor_class]) < 2:
            return self.random_train_sample()
        # Randomly choose an anchor
        anchor = random.choice(self.knwb[anchor_class])
        # Randomly choose a positive sample, ensuring it's different from the anchor
        positive = random.choice(self.knwb[anchor_class])
        while torch.equal(positive, anchor):
            positive = random.choice(self.knwb[anchor_class])
        # Randomly choose a different category for the negative sample
        negative_class = random.choice([x for x in standard_question_index if x != anchor_class])
        # Randomly choose a negative sample
        negative = random.choice(self.knwb[negative_class])
        # Return the triplet (anchor, positive, negative)
        return [anchor, positive, negative]

    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # Start indexing from 1, reserving 0 for padding
        return token_dict

    def load_schema(self, schema_path):
        with open(schema_path, encoding="utf8") as f:
            return json.loads(f.read())

def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl




if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("valid_tag_news.json", Config)
    print(dg[1])
