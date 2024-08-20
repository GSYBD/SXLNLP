import json
from encode_util import EncodeUtil
from torch.utils.data import DataLoader

import torch


class DataGenerator:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.vocab = load_vocab(config['vocab_path'])
        self.config['vocab_size'] = len(self.vocab)
        self.schema = load_schema(config['schema_path'])
        self.config['class_num'] = len(self.schema)
        self.sentences = []
        self.use_bert = config['use_bert']
        self.encodeUtil = EncodeUtil(self.use_bert, self.vocab, max_length=config['max_length'],
                                     bert_path=config['pretrain_model_path'])
        self.load()

    def load(self):
        self.data = []
        with open(self.data_path, encoding='utf8') as f:
            for line in f:
                if len(line) > self.config['max_length']:
                    for index in range(len(line) // self.config['max_length']):
                        text = line[index * self.config['max_length']: (index + 1) * self.config['max_length']]
                        self.process_sentence(text)
                else:
                    self.process_sentence(line)

    def process_sentence(self, text):
        sentence = []
        label = []
        for index, char in enumerate(text[:-1]):
            if char in self.schema:
                continue
            sentence.append(char)
            next_char = text[index + 1]
            if next_char in self.schema:
                label.append(self.schema[next_char])
            else:
                label.append(0)
        sentence = "".join(sentence)
        self.sentences.append(sentence)
        if self.use_bert:
            input_id, attention_mask = self.encodeUtil.encode_sentence(sentence)
        label = self.encodeUtil.padding(label, pad_token=-1)
        self.data.append([torch.LongTensor(input_id), torch.LongTensor(attention_mask), torch.LongTensor(label)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


def load_schema(schema_path):
    with open(schema_path, encoding='utf8') as f:
        return json.loads(f.read())


def load_data(data_path, config, shuffle=True):
    ds = DataGenerator(data_path, config)
    dl = DataLoader(ds, shuffle=shuffle, batch_size=config['batch_size'])
    return dl


if __name__ == '__main__':
    from config import Config

    ds = DataGenerator(Config['train_data_path'], Config)
    print(ds[0])
