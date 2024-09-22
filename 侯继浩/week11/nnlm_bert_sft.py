#coding:utf8
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

"""
基于pytorch的LSTM语言模型
"""

class DataGenerator:
    def __init__(self, data_path):
        self.path = data_path
        self.sentences = []
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
        self.max_length = 20
        self.load()

    def load(self):
        self.data = []
        sep_id = self.tokenizer.sep_token_id
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                question = random.choice(line["questions"])
                target = line["target"]
                question_id = self.tokenizer.encode(question, add_special_tokens=True)
                target_id = self.tokenizer.encode(target, add_special_tokens=False)
                input_id = question_id+target_id+[sep_id]
                labels = question_id[:-1]+target_id+[sep_id]
                input_id = self.padding(input_id)
                labels = self.padding(labels)
                mask = torch.ones((len(input_id), len(input_id)))
                for i in range(len(input_id)):
                    for j in range(len(input_id)):
                        if i < len(question_id)-1:
                            if j >= len(question_id)-1:
                                mask[i][j] = 0
                        else:
                            if j > i:
                                mask[i][j] = 0
                self.data.append([torch.LongTensor(input_id), torch.LongTensor(labels), mask])
        return

    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.max_length]
        input_id += [pad_token] * (self.max_length - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

#用torch自带的DataLoader类封装数据
def load_data(data_path, batch_size, shuffle=True):
    dg = DataGenerator(data_path)
    dl = DataLoader(dg, batch_size=batch_size, shuffle=shuffle)
    return dl


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, char_dim):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained('./bert-base-chinese', return_dict=False)
        self.classify = nn.Linear(hidden_size, char_dim)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, mask=None):
        if y is not None:
            x, _ = self.bert(x, attention_mask=mask)  # output shape:(batch_size, sen_len, input_dim)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)

tokenizers = BertTokenizer.from_pretrained("./bert-base-chinese")

#建立模型
def build_model(hidden_size, char_dim):
    model = LanguageModel(hidden_size, char_dim)
    return model

#文本生成测试代码
def generate_sentence(openings, model):
    openings = tokenizers.encode(openings)
    model.eval()
    with torch.no_grad():
        #生成了换行符，或生成文本超过30字则终止迭代
        while len(openings) <= 30:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)
    return tokenizers.decode(openings)

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


#计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 64       #每次训练样本个数
    train_sample = 50000   #每轮训练总共训练的样本总数
    hidden_size = 768        #每个字的维度
    vocab_size = 21128
    window_size = 10       #样本文本长度
    train_data = load_data(corpus_path, batch_size)     #加载语料
    model = build_model(hidden_size, vocab_size)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for idex,data in enumerate(train_data):
            x, y, mask = data
            if torch.cuda.is_available():
                x, y, mask = x.cuda(), y.cuda(), mask.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y, mask)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("train.json", False)
