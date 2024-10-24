#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertTokenizer

"""
基于bert语言模型生成文本
"""


class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        #self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        self.tokenizer = BertTokenizer.from_pretrained(r"C:\Users\86358\Desktop\nlp\课件\week6 语言模型\bert-base-chinese")
        self.encoder = BertModel.from_pretrained(r"C:\Users\86358\Desktop\nlp\课件\week6 语言模型\bert-base-chinese", return_dict=False)
        hidden_size = self.encoder.config.hidden_size
        vocab_size = self.encoder.config.vocab_size
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if y is not None:
            mask = torch.tril(torch.ones(x.shape[0], x.shape[1], x.shape[1]))
            x, _ = self.encoder(x, attention_mask=mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x = self.encoder(x)[0]
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)


#加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
    return vocab

#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer,window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位
    # print(window, target)
    x = tokenizer.encode(window, add_special_tokens=False, padding='max_length', truncation=True, max_length=15)
    y = tokenizer.encode(target, add_special_tokens=False, padding='max_length', truncation=True, max_length=15)
    return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(tokenizer,sample_length, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer,window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model():
    model = LanguageModel()
    return model

#文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
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
    epoch_num = 20        #训练轮数
    batch_size = 64       #每次训练样本个数
    train_sample = 50000   #每轮训练总共训练的样本总数
    window_size = 10       #样本文本长度
    vocab = build_vocab(r"C:\Users\86358\Desktop\nlp\课件\week6 语言模型\bert-base-chinese\vocab.txt")       #建立字表
    tokenizer = BertTokenizer.from_pretrained(r"C:\Users\86358\Desktop\nlp\课件\week6 语言模型\bert-base-chinese")
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model()    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(tokenizer,batch_size, window_size, corpus) #构建一组训练样本
            print(x.shape, y.shape)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            print(x.shape, y.shape)
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", False)
