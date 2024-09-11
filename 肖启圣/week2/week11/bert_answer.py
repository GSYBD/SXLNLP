# coding:utf8
import json

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import BertModel, BertTokenizer
from torch.utils.data import dataset, DataLoader

"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, input_dim, vocab_size, pretained_path):
        super(LanguageModel, self).__init__()

        self.bert = BertModel.from_pretrained(pretained_path, return_dict=False)
        self.classify = nn.Linear(input_dim, vocab_size)

        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, mask, y=None):

        x, _ = self.bert(input_ids=x, attention_mask=mask)  # output shape:(batch_size, sen_len, input_dim)
        y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
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
    with open(path, encoding="utf8") as f:
        for line in f:
            line = json.load(line)
            corpus.append([line["title"], line["content"]])
    return corpus


def berttokenizer(pretained_path):
    tokenizers = BertTokenizer.from_pretrained(pretained_path)
    return tokenizers


# 建立数据集
# sample_length 输入需要的样本数量。需要多少生成多少
# vocab 词表
# window_size 样本长度
# corpus 语料字符串
def build_dataset(tokenizer, corpus, max_length, batch_size):
    dataset = []
    for title, content in enumerate(corpus):
        title_encode = tokenizer.encode(title, add_special_tokens=False)
        content_encode = tokenizer.encode(content, add_special_tokens=False)
        x = [tokenizer.cls_token_id] + title_encode + [tokenizer.sep_token_id] + content_encode + [
            tokenizer.sep_token_id]
        y = len(title_encode) * [-1] + [-1] + content_encode + [tokenizer.sep_token_id] + [-1]

        mask = create_mask(len(title_encode), len(content_encode))

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
    # 创建掩码张量[1,1,1,0,0
    #           1,1,1,0,0
    #           1,1,1,0,0
    #           1,1,1,1,0
    #           1,1,1,1,1,]
    # 分为4个part
    # part1:s1*s1
    mask1 = torch.ones(len_s1, len_s1)
    # part2:s1*s2
    mask2 = torch.zeros(len_s1, len_s2)
    # part3:s2*s1
    mask3 = torch.ones(len_s2, len_s1)
    # part4:s2*s2
    mask4 = torch.tril(torch.ones(len_s2, len()), diagonal=0)
    # 拼接矩阵
    mask = torch.cat([torch.cat([mask1, mask2], dim=1), torch.cat([mask3, mask4], dim=1)], dim=0)
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


# 文本生成测试代码


def generate_sentence(openings, model, tokenizers):
    model.eval()
    openings = tokenizers.encode(openings)
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过30字则终止迭代
        while len(openings) <= 50:

            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)
    return tokenizers.decoce(openings)


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
    batch_size = 64  # 每次训练样本个数

    char_dim = 768  # 每个字的维度
    max_length = 50  # 样本文本长度
    vocab_size = 21128
    pretained_path = r"F:\Python Learn\Learning data\Week6 预训练模型\bert-base-chinese"
    tokenizers = berttokenizer(pretained_path)
    corpus = load_corpus(corpus_path)  # 加载语料
    model = LanguageModel(char_dim, vocab_size, pretained_path)  # 建立模型
    train_data = build_dataset(tokenizers, corpus, max_length, batch_size)  # 构建一组训练样本
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, mask, y in enumerate(train_data):
            if torch.cuda.is_available():
                x, mask, y = x.cuda(), mask.cuada(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, mask, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("北京明年拟推工作日半价观看电影", model, tokenizers))
        print(generate_sentence("南京一合金厂锅炉发生爆炸", model, tokenizers))
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