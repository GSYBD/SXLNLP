import json
import random
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
class DataGenerator:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.data_type = None

        self.vocab = load_vocab(config['vocab_path'])
        self.config['vocab_size'] = len(self.vocab)
        self.schema = load_schema(config['schema_path'])

        self.use_bert = config['use_bert']
        if self.use_bert:
            self.tokenizer = BertTokenizer.from_pretrained(config['pretrain_model_path'])
        self.load()

    def load(self):
        self.data = []
        self.know = defaultdict(list)
        with open(self.data_path, encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                if isinstance(line, dict):
                    self.data_type = 'train'
                    label = line['target']
                    label_index = self.schema[label]
                    questions = line['questions']
                    for question in questions:
                        if self.use_bert:
                            input_id = self.tokenizer.encode(question, max_length=self.config['max_length'],
                                                             pad_to_max_length=True)
                        else:
                            input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.know[label_index].append(input_id)

                else:
                    assert isinstance(line, list)
                    self.data_type = 'test'
                    question, label = line
                    label_index = self.schema[label]
                    label_index = torch.LongTensor([label_index])
                    if self.use_bert:
                        input_id = self.tokenizer.encode(question, max_length=self.config['max_length'], pad_to_max_length=True)
                    else:
                        input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    self.data.append([input_id, label_index])

    def encode_sentence(self, text):
        input_id = [self.vocab.get(c, self.vocab['[UNK]']) for c in text]
        return self.padding(input_id)

    def padding(self, input_id):
        input_id = input_id[:self.config['max_length']]
        input_id += [0] * (self.config['max_length'] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == 'train':
            return self.config['epoch_data_size']
        else:
            return len(self.data)

    def __getitem__(self, item):
        if self.data_type == 'train':
            return self.random_sample()
        else:
            return self.data[item]

    def random_sample(self):
        stander_index = list(self.know.keys())

        i1, i2 = random.sample(stander_index, 2)
        if len(self.know[i1]) < 2:
            s1 = self.know[i1][0]
            s2 = self.know[i1][0]
        else:
            s1, s2 = random.sample(self.know[i1], 2)
        s3 = random.choice(self.know[i2])
        return [s1, s2, s3]

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
    dl = DataLoader(ds, shuffle=shuffle, batch_size=config['batch_size'])
    return dl

if __name__ == '__main__':
    from config import Config

    ds = DataGenerator(Config['train_data_path'], Config)
    print(ds[0])

