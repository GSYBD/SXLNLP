#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import BertModel
import re
import json

"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, vocab):
        super(LanguageModel, self).__init__()
        self.bert_layer=BertModel.from_pretrained(r"../bert-base-chinese",
                                                  return_dict=False)
        input_dim=self.bert_layer.config.hidden_size
        self.classify = nn.Linear(input_dim, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None,mask=None):
        if y is not None:
            x,_=self.bert_layer(x, attention_mask=mask)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1),ignore_index=0)
        else:
            x,_=self.bert_layer(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)

#加载字表
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
    return vocab

#加载语料
def load_corpus(path):
    corpus = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            corpus.append([data["title"],data["content"]])
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位
    # print(window, target)
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]   #将字转换成序号
    y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    return x, y

# <sep>,<eso>
#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串

def gen_mask(len_q,max_length):
    a=len_q+1
    b=max_length-a
    upper_left=torch.ones((a,a))
    upper_right=torch.zeros((a,b))
    lower_left=torch.ones((b,a))
    lower_right=torch.tril(torch.ones((b,b)))
    upper_half = torch.cat((upper_left, upper_right), dim=1)
    lower_half = torch.cat((lower_left, lower_right), dim=1)
    mask = torch.cat((upper_half, lower_half), dim=0)
    return mask




def build_dataset(sample_nums,path,max_length,vocab):
    dataset_x = []
    dataset_y = []
    masks=[]
    corpus=load_corpus(path)
    for _ in range(sample_nums):
        sequence=corpus[random.randint(0,len(corpus)-1)]
        len_q=len(sequence[0])
        s=[]
        for char in sequence[0]:
            s.append(vocab.get(char, vocab["[UNK]"]))
        s.append(vocab.get("[SEP]"))
        for char in sequence[1]:
            s.append(vocab.get(char, vocab["[UNK]"]))
        s=s[:max_length]
        s.append(vocab.get("[ESO]"))
        s=s+[0]*(max_length+1-len(s))
        x=s[:-1]
        y=s[1:]
        y[:len_q+1]=[0]*(len_q+1)
        dataset_x.append(x)
        dataset_y.append(y)
        mask=gen_mask(len_q,max_length)
        masks.append(mask)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y),torch.stack(masks)

#建立模型
def build_model(vocab):
    model = LanguageModel(vocab)
    return model

#文本生成测试代码
def generate_sentence(openings, model, vocab, max_length):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "[ESO]" and len(openings) <= 300:
            openings += pred_char
            x = [vocab.get(char, vocab["[UNK]"]) for char in openings[-max_length:]]
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
    return openings

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
    epoch_num = 100       #训练轮数
    batch_size = 32       #每次训练样本个数
    vocab = build_vocab(r"../bert-base-chinese/vocab.txt",
                                      return_dict=False)#建立字表
    corpus_path="sample_data.json"
    max_length=256
    model = build_model(vocab)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.00001)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(100):
            x, y, mask= build_dataset(batch_size,corpus_path,max_length,vocab)
            if torch.cuda.is_available():
                x, y, mask= x.cuda(), y.cuda(), mask.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y,mask)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("阿根廷歹徒抢服装尺码不对拿回店里换", model, vocab, max_length))
        print(generate_sentence("国际通用航空大会沈阳飞行家表演队一飞机发生坠机，伤亡不明", model, vocab, max_length))
    if not save_weight:
        return
    else:
        torch.save(model.state_dict(), "model.pth")
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", False)

