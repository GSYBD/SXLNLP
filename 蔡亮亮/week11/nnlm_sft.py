import json

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re

from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(r"E:\BaiduNetdiskDownload\第六周 预训练模型\bert-base-chinese", return_dict=False)
        self.classify = nn.Linear(input_dim, len(vocab))
        # self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, attention_mask=None):
        # x,_ = self.bert.forward(x, attention_mask=attention_mask)
        # y_pred = self.classify(x)  # output shape:(batch _size, vocab_size)
        if y is not None:
            x, _ = self.bert(x, attention_mask=attention_mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)

 # 加载字表
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index + 1  # 留出0位给pad token
    return vocab


# 加载语料
def load_corpus(path):
    title = []
    content = []
    with open(path, encoding="utf8") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            title.append(line["title"])
            content.append(line["content"])
    return [title, content]



# sft的数据构造
def build_dataset(tokenizer, corpus, max_length, batch_size):
    dataset = []
    for i, (prompt, answer) in enumerate(corpus):
        prompt_encode = tokenizer.encode(prompt, add_special_tokens=False)
        answer_encode = tokenizer.encode(answer, add_special_tokens=False)
        y = len(prompt_encode) * [-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
        x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [tokenizer.sep_token_id]
        mask = create_mask(len(prompt_encode), len(answer_encode))
        # padding
        x = x[:max_length] + [0] * (max_length - len(x))
        y = y[:max_length] + [0] * (max_length - len(y))
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        mask = pad_mask(mask, (max_length, max_length))
        dataset.append([x, mask, y])

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

def create_mask(s1, s2):
    len_s1 = s1 + 2  # cls + sep
    len_s2 = s2 + 1  # sep
    # 创建掩码张量
    mask = torch.ones(len_s1 + len_s2, len_s1 + len_s2)
    # 遍历s1的每个token
    for i in range(len_s1):
        # s1的当前token不能看到s2的任何token
        mask[i, len_s1:] = 0
        # 遍历s2的每个token
    for i in range(len_s2):
        # s2的当前token不能看到后面的s2 token
        mask[len_s1 + i, len_s1 + i + 1:] = 0
    return mask


def pad_mask(tensor, target_shape):
    # 获取输入张量和目标形状的长宽
    height, width = tensor.shape
    target_height, target_width = target_shape
    # 创建一个全零张量,形状为目标形状
    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    # 计算需要填充或截断的区域
    h_start = 0
    w_start = 0
    h_end = min(height, target_height)
    w_end = min(width, target_width)
    # 将原始张量对应的部分填充到全零张量中
    result[h_start:h_end, w_start:w_end] = tensor[:h_end - h_start, :w_end - w_start]
    return result


# 建立模型
def build_model(vocab, char_dim):
    model = LanguageModel(char_dim, vocab)
    return model

# 文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        # 生成文本超过30字则终止迭代
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
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


def main(corpus_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    char_dim = 768  # 每个字的维度
    max_length = 50
    # vocab = build_vocab("vocab.txt")       #建立字表
    vocab = build_vocab(r"E:\BaiduNetdiskDownload\第六周 预训练模型\bert-base-chinese\vocab.txt")
    corpus = load_corpus(corpus_path)  # 加载语料
    model = build_model(vocab, char_dim)  # 建立模型
    tokenizer = BertTokenizer.from_pretrained(r"E:\BaiduNetdiskDownload\第六周 预训练模型\bert-base-chinese")
    train_data = build_dataset(tokenizer, corpus, max_length, batch_size)  # 建立数据集
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, y, mask in train_data:  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y, mask= x.cuda(), y.cuda(), mask.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, mask, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("明确指出学习雷锋精神", model, tokenizer))
        print(generate_sentence("朝鲜16日宣布将于", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    main(r"D:\JetBrainsProject\PyCharm\NLPLearn\tenWeek\transformers-生成文章标题\sample_data.json", False)