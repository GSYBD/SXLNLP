import json
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

'''
加载训练数据
'''


class DataGenerator:
    def __init__(self, data_path, config):
        self.data_path = data_path
        self.config = config
        self.schema = load_schema(config['schema_path'])
        self.config['class_num'] = len(self.schema)
        self.tokenizer = load_vocab(config['vocab_path'])
        self.config['vocab_size'] = len(self.tokenizer.vocab)
        self.max_len = config['max_length']
        self.load()

    def load(self):
        self.data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n\n')

            for line in lines:
                text = []
                labels = [8]
                line_data = line.split('\n')
                for data in line_data:
                    data = data.strip()
                    if data == '':
                        continue
                    char, label = data.split()
                    text.append(char)
                    labels.append(self.schema[label])
                # 文本转序列
                text = ''.join(text)
                input_seq = text_to_seq(self.tokenizer, text,self.max_len)
                label_seq = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_seq), torch.LongTensor(label_seq)])
            # assert len(text_seq) == len(labels), print(len(text_seq), len(labels))
            # for i in range(len(text_seq) // self.max_len):
            #     input_seq = text_seq[i * self.max_len: (i + 1) * self.max_len]
            #     label_seq = [8]+labels[i * (self.max_len-1): (i + 1) * (self.max_len-1)]
            #     if i == len(text_seq) // self.max_len - 1:
            #         input_seq = self.padding(input_seq)
            #         label_seq = self.padding(label_seq, -1)
            #     self.data.append([torch.LongTensor(input_seq), torch.LongTensor(label_seq)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def padding(self, seq, padding_token=0):
        seq = seq[:self.max_len]
        seq = seq + [padding_token] * (self.max_len - len(seq))
        return seq


def load_schema(schema_path):
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    return schema


# 文本转序列
# def text_to_seq(text, vocab):
#     text_seq = []
#     for word in text:
#         text_seq.append(vocab.get(word, vocab['<unk>']))
#     return text_seq

def text_to_seq(tokenizer, text,max_len):
    text_seq = tokenizer.encode(text,padding = 'max_length',max_length=max_len,truncation=True)
    return text_seq


# def load_vocab(vocab_path):
#     vocab = {}
#     with open(vocab_path, 'r', encoding='utf-8') as f:
#         for index, line in enumerate(f):
#             char = line.strip()
#             vocab[char] = index + 1
#         vocab['<unk>'] = len(vocab)
#     return vocab

# 使用bert的tokenizer
def load_vocab(vocab_path):
    tokenizer = BertTokenizer(vocab_path)
    return tokenizer


def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config['batch_size'], shuffle=shuffle)
    return dl


if __name__ == '__main__':
    from config import Config

    dl = load_data('ner_data/train.txt', Config)
    print(dl.__len__())
