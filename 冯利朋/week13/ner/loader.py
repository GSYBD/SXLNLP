import json

import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
class DataGenerator:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        # 加载schema
        self.schema = load_schema(config['schema_path'])
        self.config['class_num'] = len(self.schema)

        # 加载字典
        self.vocab = load_vocab(config['vocab_path'])
        self.config['vocab_size'] = len(self.vocab)

        # 是否使用bert
        self.use_bert = config['use_bert']
        if self.use_bert:
            self.tokenizer = BertTokenizer.from_pretrained(config['pretrain_model_path'])

        self.sentences = []
        # 加载数据
        self.load_data()

    def load_data(self):
        self.data = []
        with open(self.data_path, encoding='utf8') as f:
            # 拆分出文本的每一句话
            segments = f.read().split("\n\n")
            # 再拆分每句话的每个字
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentence))
                if self.use_bert:
                    input_id = self.tokenizer.encode(sentence, max_length=self.config['max_length'], pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(sentence)
                labels = self.padding(labels, pad_token=-1)
                self.data.append([torch.LongTensor(input_id), torch.LongTensor(labels)])
    def encode_sentence(self, text, padding=True):
        sequence = [self.vocab.get(c, self.vocab['[UNK]']) for c in text]
        if padding:
            return self.padding(sequence)
        return sequence

    def padding(self, input_id, pad_token = 0):
        input_id = input_id[:self.config['max_length']]
        input_id += [pad_token] * (self.config['max_length'] - len(input_id))
        return input_id
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def load_data(data_path, config):
    ds = DataGenerator(data_path, config)
    dl = DataLoader(ds, shuffle=True, batch_size=config['batch_size'])
    return dl

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict
def load_schema(schema_path):
    with open(schema_path, encoding='utf8') as f:
        return json.loads(f.read())

if __name__ == '__main__':
    from config import Config
    ds = DataGenerator(Config['train_data_path'], Config)
    print(ds[0])