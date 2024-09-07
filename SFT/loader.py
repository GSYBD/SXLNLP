import json
import numpy as np
import torch
from transformers import BertTokenizer

from config import Config


class DataGenerator:
    def __init__(self, data_path):
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(Config['bert'])
        self.dataset = []
        self.load()

    def load(self):
        with open(self.data_path, encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)  # {'title': str, 'content': str, ..., 'title': str, 'content': str}

                # question = self.tokenizer.encode(line['title'])
                # answer = self.tokenizer.encode(line['content'])[1:]

                text = line['title'] + '[SEP]' + line['content']

                text = self.tokenizer.encode(text,
                                             padding='max_length',
                                             max_length=Config['max_length'],
                                             truncation=True)

                # text = padding(text)

                # print(text)  # [101, int, ..., 102, int, ..., int, 102]

                mid = text.index(102)

                if mid == 63:
                    print(mid)

                question = text[:mid + 1]
                answer = text[mid + 1:]

                mask_left = np.ones((len(text), len(question)))
                mask_right_up = np.zeros((len(question), len(answer)))
                mask_right_down = np.tril(np.ones((len(answer), len(answer))))
                mask_right = np.concatenate((mask_right_up, mask_right_down), axis=0)
                mask = np.concatenate((mask_left, mask_right), axis=1)  # (64, 64)

                x = text
                x = torch.LongTensor(x)

                no_padding_answer = [integer for integer in answer if integer != 0]

                if len(no_padding_answer) == len(answer):
                    y = (len(question) - 1) * [-1] + answer + [-1]
                    y = torch.LongTensor(y)
                    self.dataset.append([x, mask, y])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


# def padding(text, pad_token=0):
#     text = text[:Config['max_length']]
#     text += [pad_token] * (Config['max_length'] - len(text))
#     return text
