# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer
from transformers import BertModel

"""
bert做自回归

"""

bert_path = "/Users/zuiqingfeng/Documents/八斗人工智能/第六周 预训练模型/bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(bert_path)


class LanguageModel(nn.Module):
    def __init__(self, vocab):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path, return_dict=False)

        self.classify = nn.Linear(self.bert.config.hidden_size, len(vocab))
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, mask=None):
        """
        x: [64, 10]
        y: [64, 10]

        ypred: [64, 10, 3961]

        mask.shape [64, 10, 10]

        :param x:
        :param y:
        :return:
        """
        x,_ = self.bert(x,attention_mask=mask)  # output shape:(batch_size, sen_len, input_dim)

        y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)

        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


# 加载字表
def build_vocab():
    vocab = tokenizer.get_vocab()
    return vocab


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(vocab, window_size, corpus):

    start = random.randint(0, len(corpus) - 1 - window_size)

    end = start + window_size
    window = corpus[start:end]

    target = corpus[start + 1:end + 1]  # 输入输出错开一位

    x = tokenizer.encode(window, max_length=window_size, padding='max_length', truncation=True)

    mask = 1 - torch.triu(torch.ones(window_size, window_size,dtype=torch.int64))
    mask = mask + torch.eye(window_size)

    y = tokenizer.encode(target, max_length=window_size, padding='max_length', truncation=True)
    return x, y, mask.tolist()


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    dataset_mask = []
    for i in range(sample_length):
        x, y, mask = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
        dataset_mask.append(mask)

    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y),torch.tensor(dataset_mask,dtype=torch.int64)


# 建立模型
def build_model(vocab):
    model = LanguageModel(vocab)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            print(x)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            # 这里模型的输出是 bath——size，句子长度 * 词表长度
            # 因为是用上一个字预测下一个字，所以这里实际上是只需要最后一个字的维度，来找下一个字
            # 这里是不会有终止的，是个简单的模型
            # print(model(x),model(x).shape)
            y = model(x)[0][-2]

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
    return 2 ** (prob * (-1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数

    window_size = 10  # 样本文本长度
    vocab = build_vocab()  # 建立字表
    corpus = load_corpus(corpus_path)  # 加载语料
    model = build_model(vocab)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y, mask = build_dataset(batch_size, vocab, window_size, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, y, mask)  # 计算loss
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
