# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re

from transformers import BertModel, BertTokenizer

"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        self.encoder = BertModel.from_pretrained(r"E:\ai课程\八斗精品班\week6 语言模型和预训练\bert-base-chinese", return_dict=False)
        self.tokenizer = BertTokenizer.from_pretrained(r"E:\ai课程\八斗精品班\week6 语言模型和预训练\bert-base-chinese")
        self.max_length = self.encoder.config.max_position_embeddings
        input_dim = self.encoder.config.hidden_size
        self.classify = nn.Linear(input_dim, len(vocab))
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy
    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, mask=None):
        # x = self.tokenizer.encode(x, max_length=self.max_length,padding="max_langth",
        #                           pad_to_max_length=True)  # output shape:(batch_size, sen_len, input_dim)
        if mask is not None:
            x, _ = self.encoder(x, attention_mask=mask)
        else:
            x, _ = self.encoder(x)  # output shape:(batch_size, sen_len, input_dim)
        y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
        # print(y_pred.shape)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


# 加载字表
def build_vocab(vocab_path):
    # vocab = {"<pad>": 0}
    vocab = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index
    return vocab


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

def get_mask(batch_size,max_length=15,window_size=10):
    batch_mask = []
    shape = (max_length, max_length)
    mask = torch.full(shape, 1e-9)
    for i in range(window_size+1):
        for j in range(i+1):
            mask[i,j] = 1
    mask = mask.tolist()
    for _ in range(batch_size):
        batch_mask.append(mask)
    return torch.LongTensor(batch_mask)
# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    x = tokenizer.encode(window,max_length=15, padding="max_length",
                                 pad_to_max_length=True)  # output shape:(sen_len, input_dim)
    y = tokenizer.encode(target,max_length=15, padding="max_length",
                                 pad_to_max_length=True)  # output shape:( sen_len, input_dim)
    # print(window, target)
    # x = [vocab.get(word, vocab["[UNK]"]) for word in window]  # 将字转换成序号
    # y = [vocab.get(word, vocab["[UNK]"]) for word in target]
    return x, y


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(tokenizer, sample_length, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    tokenizer = model.tokenizer
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            # x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            str = openings[-window_size:]
            x = tokenizer.encode(str,max_length=15, padding="max_length",
                                 pad_to_max_length=True)  # output shape:(bsen_len, input_dim)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)
            y = y[0][-1]
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


# 计算文本ppl
def calc_perplexity(tokenizer, sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            # x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            # x = torch.LongTensor([x])
            x = tokenizer.encode(window, max_length=15, padding="max_length",
                                 pad_to_max_length=True)  # output shape:( sen_len, input_dim)
            x = torch.LongTensor(x)
            target = sentence[i]
            target_index = vocab.get(target, vocab["[UNK]"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * (-1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    char_dim = 768  # 每个字的维度
    window_size = 10  # 样本文本长度
    vocab = build_vocab(r"E:\ai课程\八斗精品班\week6 语言模型和预训练\bert-base-chinese\vocab.txt")  # 建立字表
    corpus = load_corpus(corpus_path)  # 加载语料
    mask = get_mask(batch_size)
    model = build_model(vocab, char_dim)  # 建立模型
    tokenizer = model.tokenizer
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(tokenizer, batch_size, window_size, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, y=y, mask=mask)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
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
    # t = [[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]]
    # print(t[0][-1])
    # str = "123456"
    # print(str[-2:])