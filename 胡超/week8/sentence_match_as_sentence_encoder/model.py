# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import abc

"""
建立网络模型结构
"""


class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        # max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    # 输入为问题字符编码
    def forward(self, x):
        x = self.embedding(x)
        # 使用lstm
        # x, _ = self.lstm(x)
        # 使用线性层
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


class SiameseNetwork(nn.Module, abc.ABC):
    def __init__(self, config):
        super().__init__()
        self.sentence_encoder = SentenceEncoder(config)

    @property
    @abc.abstractmethod
    def loss(self):
        pass

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    @staticmethod
    def cosine_distance(tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), dim=-1)
        return 1 - cosine

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass


class SiameseNetworkCEL(SiameseNetwork):
    def __init__(self, config):
        super().__init__(config)

    @property
    def loss(self):
        return nn.CosineEmbeddingLoss()

    # sentence : (batch_size, max_length)
    def forward(self, sentence1, sentence2=None, target=None):
        # 同时传入两个句子
        if sentence2 is not None:
            vector1 = self.sentence_encoder(sentence1)  # vec:(batch_size, hidden_size)
            vector2 = self.sentence_encoder(sentence2)
            # 如果有标签，则计算loss
            if target is not None:
                return self.loss(vector1, vector2, target.squeeze())
            # 如果无标签，计算余弦距离
            else:
                return self.cosine_distance(vector1, vector2)
        # 单独传入一个句子时，认为正在使用向量化能力
        else:
            return self.sentence_encoder(sentence1)


class SiameseNetworkTriplet(SiameseNetwork):
    def __init__(self, config):
        super().__init__(config)

    @property
    def loss(self):
        return self.cosine_triplet_loss

    def cosine_triplet_loss(self, a, p, n, margin=0.1):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if type(margin) != float or margin <= 0:
            raise ValueError("margin must be float and greater than 0")
        diff = ap - an + margin
        return torch.mean(diff[diff.gt(0)])  # greater than

    def forward(self, a, p=None, n=None, margin=0.1):
        if p is not None and n is not None:
            vector_a = self.sentence_encoder(a)
            vector_p = self.sentence_encoder(p)
            vector_n = self.sentence_encoder(n)
            return self.cosine_triplet_loss(vector_a, vector_p, vector_n, margin)
        else:
            return self.sentence_encoder(a)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config

    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model_test = SiameseNetworkCEL(Config)
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    ll = torch.LongTensor([[1], [0]])
    y = model_test(s1, s2, ll)
    print(y)
    # print(model.state_dict())
