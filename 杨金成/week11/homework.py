# @Version  : 1.0
# @Author   : acyang
# @File     : homework.py
# @Time     : 2024/9/5 15:10
# coding:utf8
import json

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader

"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)

        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, mask=None, y=None):
        if y is not None:
            # 训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            # mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            if torch.cuda.is_available() and mask is not None:
                mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)


# 加载字表
# def build_vocab(vocab_path):
#     vocab = {"<pad>":0}
#     with open(vocab_path, encoding="utf8") as f:
#         for index, line in enumerate(f):
#             char = line[:-1]       #去掉结尾换行符
#             vocab[char] = index + 1 #留出0位给pad token
#     return vocab

# 加载语料
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            corpus.append([line['title'], line['content']])
    return corpus


# sft的数据构造
def build_dataset(tokenizer, corpus, max_len, batch_size):
    dataset = []
    for i, (prompt, answer) in enumerate(corpus):
        prompt_encode = tokenizer.encode(prompt, add_special_tokens=False)
        answer_encode = tokenizer.encode(answer, add_special_tokens=False)
        x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [
            tokenizer.sep_token_id]
        y = len(prompt_encode) * [-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
        mask = create_mask(len(prompt_encode), len(answer_encode))
        x = x[:max_len] + (max_len - len(x)) * [0]
        y = y[:max_len] + (max_len - len(x)) * [0]
        x = torch.tensor(x)
        y = torch.tensor(y)
        mask = mask_pad(mask, (max_len, max_len))
        print("--------")
        print(x.shape)
        print(mask.shape)
        print(y.shape)
        dataset.append([x, mask, y])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def create_mask(len1, len2):
    # 创建全1张量
    mask = torch.ones(len1 + len2 + 3, len1 + len2 + 3)
    for i in range(len1 + 2):
        mask[i, len1 + 2:] = 0
    for j in range(len2 + 1):
        mask[len1 + 2 + j, len1 + 2 + j + 1:] = 0
    return mask


def mask_pad(m, target_shape):
    h, w = m.shape
    target_h, target_w = target_shape
    h_end, w_end = min(h, target_h), min(w, target_w)
    mask = torch.zeros(target_shape, dtype=m.dtype, device=m.device)
    mask[:h_end, :w_end] = m[:h_end, :w_end]
    return mask


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# tokenizer 词表
# window_size 样本长度
# corpus 语料字符串
# def build_dataset(sample_length, tokenizer, window_size, corpus):
#     dataset_x = []
#     dataset_y = []
#     for i in range(sample_length):
#         x, y = build_sample(tokenizer, window_size, corpus)
#         dataset_x.append(x)
#         dataset_y.append(y)
#     return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    # reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = ''.join(tokenizer.decode(index))
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


def train(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 128  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    char_dim = 768  # 每个字的维度
    max_len = 50
    vocab_size = 21128  # 字表大小
    learning_rate = 0.001  # 学习率

    pretrain_model_path = r'../week6/bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)  # 加载语料
    model = build_model(vocab_size, char_dim, pretrain_model_path)  # 建立模型
    data = build_dataset(tokenizer, corpus, max_len, batch_size)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, mask, y in data:
            if torch.cuda.is_available():
                x, mask, y = x.cuda(), mask.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, mask, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("广州农业局长回应与王林蛇缠脖合影：只是玩游戏", model, tokenizer))
        print(generate_sentence("冰岛女总理将携夫人访华 为该国首对同性结婚伴侣", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # corpus = load_corpus('sample_data.json')
    # print(corpus)
    # print(create_mask(3, 2))
    # pretrain_model_path = r'../week6/bert-base-chinese'
    # tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    # print(build_dataset(tokenizer, corpus, 50, 1))
    train('sample_data.json', False)

