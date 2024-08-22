import json

import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader


class DataGenerator:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.vocab = load_vocab(config['vocab_path'])
        self.config['vocab_size'] = len(self.vocab)
        self.schema = load_schema(config['schema_path'])
        self.config['class_num'] = len(self.schema)
        self.use_bert = self.config['use_bert']
        self.sentences = []
        if self.use_bert:
            self.tokenizer = BertTokenizer.from_pretrained(config['pretrain_model_path'],add_special_tokens=False)
        self.load()

    def load(self):
        self.data = []
        with open(self.data_path, encoding='utf8') as f:
            # 每一句话使用 \n\n分割的
            segments = f.read().split("\n\n")
            for segment in segments:
                # 每一句话是用\n分割的每个词
                sentence_words = segment.split("\n")
                self.process_sentence(sentence_words)

    def process_sentence(self, sentence_words):
        sentence = ""
        labels = []
        for index, words in enumerate(sentence_words):
            if words.strip() == "":
                continue
            char, label = words.split()
            sentence += char
            labels.append(self.schema[label])
        if self.use_bert:
            input_id = self.tokenizer.encode(sentence, add_special_tokens=False, max_length=self.config['max_length'],
                                             padding='max_length',
                                             truncation=True)
        else:
            input_id = self.encode_sentence(sentence)
        labels = self.padding(labels, pad_token=-1)
        self.sentences.append(sentence)
        self.data.append([torch.LongTensor(input_id), torch.LongTensor(labels)])

    def encode_sentence(self, text):
        input_id = [self.vocab.get(c, self.vocab['[UNK]']) for c in text]
        return self.padding(input_id)

    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config['max_length']]
        input_id += [pad_token] * (self.config['max_length'] - len(input_id))
        return input_id

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def load_schema(schema_path):
    with open(schema_path, encoding='utf8') as f:
        return json.loads(f.read())


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


def load_data(data_path, config, shuffle=True):
    ds = DataGenerator(data_path, config)
    dl = DataLoader(ds, batch_size=config['batch_size'], shuffle=shuffle)
    return dl


if __name__ == '__main__':
    from config import Config

    ds = DataGenerator(Config['train_data_path'], Config)
    print(ds[0])
