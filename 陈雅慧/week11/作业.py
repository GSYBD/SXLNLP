# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel
import json

"""
用bert实现SFT的训练，主要通过加入一个attention mask实现。

"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)

        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None,mask=None):
        if y is not None and mask is not None:
            # 训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            # mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            # if torch.cuda.is_available():
            #     mask = mask.cuda()
            x,_= self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1),ignore_index=-100)
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
# def load_corpus(path):
#     corpus = ""
#     with open(path, encoding="gbk") as f:
#         for line in f:
#             corpus += line.strip()
#     return corpus

def load(path):
    data = []
    with open(path, encoding="utf8") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            query = line["title"]
            answer = line["content"]
            data.append([query,answer])
    return data


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(tokenizer, max_length, pair):
    # start = random.randint(0, len(corpus) - 1 - window_size)
    # end = start + window_size
    # window = corpus[start:end]
    # target = corpus[start + 1:end + 1]  # 输入输出错开一位
    query_ans_length = len(pair[0]) + len(pair[1])
    query=tokenizer.encode(pair[0],add_special_tokens=False)
    answer = tokenizer.encode(pair[1], add_special_tokens=False)
    x=query+[102]+answer
    if len(x) > max_length:
        x = x[:max_length]  # 截断
    else:
        x += [0] * (max_length - len(x))  # 填充
    y=len(pair[0])*[-100]+x[len(pair[0])+1:query_ans_length]+[-100]*(max_length-query_ans_length+1)
    query_one=np.ones((len(pair[0])+1,len(pair[0])+1))
    ans_zero=np.zeros((len(pair[0])+1,len(pair[1])))
    ans_one=np.ones((len(pair[1]),len(pair[0])+1))
    query_ans=np.tril(np.ones((len(pair[1]),len(pair[1]))))
    A=np.concatenate((query_one, ans_zero), axis=1)
    B=np.concatenate((ans_one, query_ans), axis=1)
    mask= np.concatenate((A, B), axis=0)

    pad_mask=np.concatenate((mask,np.zeros((query_ans_length+1,max_length-query_ans_length-1))),axis=1)#将补齐的地方置为0
    pad_mask=np.concatenate((pad_mask,np.zeros((max_length-query_ans_length-1,max_length))),axis=0)#将补齐的地方置为0

    return x, y,pad_mask


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset( tokenizer,data,max_length):
    dataset_x = []
    dataset_y = []
    attention_mask=[]
    for pair in data:
        x, y,mask = build_sample(tokenizer, max_length, pair)
        dataset_x.append(x)
        dataset_y.append(y)
        attention_mask.append(mask)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y),torch.LongTensor(attention_mask)


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
        # 生成了换行符，或生成文本超过100字则终止迭代
        while pred_char != "\n" and len(openings) <= 100:
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
    epoch_num = 50  # 训练轮数
    batch_size = 128  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    char_dim = 768  # 每个字的维度
    max_length=150
    vocab_size = 21128  # 字表大小
    learning_rate = 0.00001  # 学习率

    pretrain_model_path = r'C:\Users\CYH\Desktop\学习\课件\week6 预训练模型\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    data = load(corpus_path)  # 加载语料
    model = build_model(vocab_size, char_dim, pretrain_model_path)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y,attention_mask = build_dataset(  tokenizer,data,max_length)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, y,attention_mask)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("今天的天气怎么样", model, tokenizer))
        print(generate_sentence("什么是幸福", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("sample_data.json", False)