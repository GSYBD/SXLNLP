# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
建立网络模型结构
"""

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)  # 词嵌入层
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)  # LSTM层
        self.layer = nn.Linear(hidden_size, hidden_size)  # 线性层
        self.dropout = nn.Dropout(0.5)  # Dropout层

    # 输入为问题字符编码
    def forward(self, x):
        sentence_length = torch.sum(x.gt(0), dim=-1)  # 计算句子长度
        x = self.embedding(x)  # 嵌入层
        # 使用lstm
        # x, _ = self.layer(x)
        # 使用线性层
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()  # 最大池化
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)  # 句子编码器
        self.loss = nn.CosineEmbeddingLoss()  # 余弦嵌入损失

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)  # 归一化
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)  # 归一化
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)  # 计算余弦相似度
        return 1 - cosine  # 返回余弦距离

    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)  # 计算正样本对余弦距离
        an = self.cosine_distance(a, n)  # 计算负样本对余弦距离
        if margin is None:
            diff = ap - an + 0.1  # 计算差异
        else:
            diff = ap - an + margin.squeeze()  # 计算差异
        return torch.mean(diff[diff.gt(0)])  # 返回损失

    # sentence : (batch_size, max_length)
    def forward(self, sentence1, sentence2=None, sentence3=None):
        # 同时传入3个句子,则做tripletloss的loss计算
        if sentence2 is not None and sentence3 is not None:
            vector1 = self.sentence_encoder(sentence1)  # 编码句子1
            vector2 = self.sentence_encoder(sentence2)  # 编码句子2
            vector3 = self.sentence_encoder(sentence3)  # 编码句子3
            return self.cosine_triplet_loss(vector1, vector2, vector3)  # 计算三元组损失
        # 单独传入一个句子时，认为正在使用向量化能力
        else:
            return self.sentence_encoder(sentence1)  # 编码句子


def choose_optimizer(config, model):
    optimizer = config["optimizer"]  # 获取优化器类型
    learning_rate = config["learning_rate"]  # 获取学习率
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)  # 返回Adam优化器
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)  # 返回SGD优化器


if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)  # 创建模型
    s1 = torch.LongTensor([[1,2,3,0], [2,2,0,0]])  # 句子1
    s2 = torch.LongTensor([[1,2,3,4], [3,2,3,4]])  # 句子2
    l = torch.LongTensor([[1],[0]])  # 标签
    y = model(s1, s2, l)  # 前向传播
    print(y)  # 打印结果
    # print(model.state_dict())
