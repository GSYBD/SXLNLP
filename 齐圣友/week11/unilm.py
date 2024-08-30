#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re

import json

from transformers import BertModel, BertTokenizer

from torch.utils.data import Dataset, DataLoader

"""
基于pytorch的Bert结构，进行sft形式的训练
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab_size, model_path):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        self.bert = BertModel.from_pretrained(model_path, return_dict=False)
        self.classify = nn.Linear(input_dim, vocab_size)
        # self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, mask=None, y=None):
        # x = self.embedding(x)       #output shape:(batch_size, sen_len, input_dim)
        # x, _ = self.layer(x)        #output shape:(batch_size, sen_len, input_dim)
        if y is not None:
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)

#加载语料
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf8") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            title = line["title"]
            content = line["content"]
            corpus.append([title, content])
    return corpus

def prepare_data(title, content,window_size, tokenizer):
    input_seq1 = tokenizer.encode(title, add_special_tokens=False)
    input_seq2 = tokenizer.encode(content, add_special_tokens=False)
    x = [tokenizer.cls_token_id] + input_seq1 + [tokenizer.sep_token_id] + input_seq2 + [tokenizer.sep_token_id]
    y = len(input_seq1) * [-1] + [-1] + input_seq2 + [tokenizer.sep_token_id] + [-1]  # 输出序列
    mask = create_mask(len(input_seq1), len(input_seq2))
    x = x[:window_size] + [0]*(window_size - len(x))
    y = y[:window_size] + [0]*(window_size - len(y))
    return x, mask, y

#创建掩码，输入两个字符串的长度
def create_mask(l1, l2):
    len_s1 = l1 + 2  # cls + sep
    len_s2 = l2 + 1  # sep
    mask = torch.ones(len_s1+len_s2,len_s1+len_s2)
    for i in range(len_s1):
        mask[i,len_s1:] = 0
    for i in range(len_s2):
        mask[len_s1+i,len_s1+i+1:] = 0
    return mask


#建立数据集
#batch_size #每次训练样本个数
#corpus 语料字符串
def build_dataset(tokenizer, corpus, window_size,batch_size):
    dataset = []
    for i, (title, content) in enumerate(corpus):
        x, mask, y = prepare_data(title, content,window_size, tokenizer)
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        mask = padding_mask(mask, (window_size, window_size))
        dataset.append([x, mask, y])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

def padding_mask(tensor, target_shape):
    # 获取输入张量和目标形状的长宽
    height, width = tensor.shape
    target_height, target_width = target_shape
    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    h_start = 0
    w_start = 0
    h_end = min(height, target_height)
    w_end = min(width, target_width)
    result[h_start:h_end, w_start:w_end] = tensor[:h_end - h_start, :w_end - w_start]
    return result

#建立模型
def build_model(vocab_size, char_dim,model_path):
    model = LanguageModel(char_dim, vocab_size,model_path)
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        #生成文本超过30字则终止迭代
        while len(openings) <= 50:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)
    return tokenizer.decode(openings)

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    # 按照概率分布去采样
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
    epoch_num = 15        #训练轮数
    batch_size = 32       #每次训练样本个数
    char_dim = 768        #每个字的维度
    window_size = 50       #样本文本长度
    vocab_size = 21128
    # vocab = build_vocab("vocab.txt")       #建立字表
    model_path = r"D:\Tools\JetBrains\Tool\bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    corpus = load_corpus(corpus_path)     #加载语料
    train_data = build_dataset(tokenizer, corpus, window_size,batch_size)
    model = build_model(vocab_size, char_dim, model_path)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, mask, y in train_data:
            if torch.cuda.is_available():
                x, mask, y = x.cuda(), mask.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, mask, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    train("sample_data.json", False)

