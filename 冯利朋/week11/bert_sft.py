import json

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

pretrain_model_path = r'/Users/gonghengan/Documents/hugging-face/bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
def build_sample(title, content, max_length):
    input = title + '[SEP]' + content
    taget = content
    x = tokenizer.encode(input, max_length=max_length, padding='max_length', add_special_tokens=False, truncation=True)
    y = tokenizer.encode(taget, max_length = (max_length - len(title)), padding='max_length', add_special_tokens=False, truncation=True)
    y = [0] * len(title) + y
    mask = np.zeros((max_length, max_length), dtype=int)
    real_length = min(max_length, len(title+content) + 1)
    mask[0:real_length, 0:len(title)+1] = 1
    for i in range(len(title)+1,real_length):
        mask[i:real_length,len(title)+1:i+1] = 1
    return x, y, mask

class Dataset:
    def __init__(self, corpus_path, max_length):
        self.corpus_path = corpus_path
        self.max_length = max_length
        self.load()
    def load(self):
        self.data = []
        with open(self.corpus_path, encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                title = line['title']
                content = line['content']
                x, y , mask = build_sample(title, content, self.max_length)
                self.data.append([torch.LongTensor(x), torch.LongTensor(y), torch.LongTensor(mask)])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

def build_dataloader(corpus_path, max_length, batch_size=20):
    ds = Dataset(corpus_path, max_length)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return dl

class TorchModel(nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        self.classify = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)
    def forward(self, x, y=None, mask=None):
        if y is not None:
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)



def main():
    epoch_num = 20
    learning_rate = 1e-5
    max_length = 150
    dl = build_dataloader('./sample_data.json', max_length)
    model = TorchModel()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    if torch.cuda.is_available():
        model = model.cuda()
    for epoch in range(epoch_num):
        watch_loss = []
        model.train()
        for index, batch_data in enumerate(dl):
            if torch.cuda.is_available():
                batch_data = [c.cuda() for c in batch_data]
            x, y, mask = batch_data
            loss = model(x, y, mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        print(f"第{epoch}轮,loss={np.mean(watch_loss)}")


if __name__ == '__main__':
    main()
    # x, y, mask = build_sample('你好吗', '我很好', 10)
    # print(x)
    # print(y)
    # print(mask)