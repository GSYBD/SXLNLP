#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import json
import random
import os
import re
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
import os.path as osp

"""
基于pytorch的LSTM语言模型
use SFT to train Q&A model
"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()

        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)  # Updated to handle padding

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        if y is not None:
            # 训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1]))).to(dtype=torch.bool)
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)   # output shape:(batch_size, sequence_length, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)   # output shape:(batch_size, sequence_length, vocab_size)
            return torch.softmax(y_pred, dim=-1)


# 加载语料
def load_corpus(path):
    title_list = []
    content_list = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            title_list.append(line["title"])
            content_list.append(line["content"])
    return [title_list, content_list]


# generate a padding sample
def build_sample(tokenizer, corpus, max_len):
    title_list, content_list = corpus
    random_index = random.randint(0, len(title_list) - 1)
    
    title = title_list[random_index]
    content = content_list[random_index]
    
    x_encoded = tokenizer.encode(title, add_special_tokens=False)
    y_encoded = tokenizer.encode(content, add_special_tokens=False)
    
    pad_x = [tokenizer.cls_token_id] + x_encoded + [tokenizer.sep_token_id] + y_encoded
    pad_y = len(x_encoded) * [-100] + [-100] + y_encoded

    pad_x = pad_x[:max_len] + [tokenizer.pad_token_id] * (max_len - len(pad_x))
    pad_y = pad_y[:max_len] + [-100] * (max_len - len(pad_y))
    
    return [torch.LongTensor(pad_x), torch.LongTensor(pad_y)]


# 建立数据集
def build_dataset(sample_length, tokenizer, corpus, max_len):
    dataset = []
    for i in range(sample_length):
        dataset.append(build_sample(tokenizer, corpus, max_len))
        
    return DataLoader(dataset, batch_size=32, shuffle=True)
    
    

# 建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, pretrain_model_path)
    return model

# 文本生成测试代码
def evaluate(openings, model, tokenizer, corpus):
    model.eval()
    # 转化为input_id
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        for i in range(50):
            x = torch.LongTensor(openings).unsqueeze(0)
            if torch.cuda.is_available():
                x = x.cuda()
            prob = model(x)
            next_index = sampling_strategy(prob[0, -1])
            if next_index == tokenizer.sep_token_id:
                break
            openings.append(next_index)
            
    return tokenizer.decode(openings)

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
    epoch_num = 20        # 训练轮数
    batch_size = 128       # 每次训练样本个数
    train_sample = 10000   # 每轮训练总共训练的样本总数
    char_dim = 768        # 每个字的维度
    window_size = 10       # 样本文本长度
    vocab_size = 21128      # 字表大小
    learning_rate = 0.001  # 学习率
    max_len = 50

    pretrain_model_path = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)     # 加载语料
    model = build_model(vocab_size, char_dim, pretrain_model_path)    # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   # 建立优化器
    dataset = build_dataset(train_sample, tokenizer, corpus, max_len)
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, y in dataset:
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    # 梯度归零
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print(f"Epoch {epoch} loss: {np.mean(watch_loss)}")
        result1 = evaluate("阿根廷歹徒抢服装尺码不对拿回店里换", model, tokenizer, corpus)
        print(f"Result1: {result1}")
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    current_dir = osp.dirname(__file__)
    train(osp.join(current_dir, "sample_data.json"), False)
