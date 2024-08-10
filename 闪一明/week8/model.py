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
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)  # 新增隐藏层
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).float()
        x = nn.functional.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.layer2(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)

    # 计算余弦距离
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def cosine_triplet_loss(self, anchor, positive, negative, margin=0.1):
        ap = self.cosine_distance(anchor, positive)
        an = self.cosine_distance(anchor, negative)
        diff = ap - an + margin

        print(f"ap shape: {ap.shape}, an shape: {an.shape}, diff shape: {diff.shape}")  # 新增调试信息

        loss = torch.mean(diff[diff.gt(0)])
        return loss

    def forward(self, sentence1, sentence2=None):
        if sentence2 is not None:
            vector1 = self.sentence_encoder(sentence1)  # vec:(batch_size, hidden_size)
            vector2 = self.sentence_encoder(sentence2)
            return vector1, vector2
        else:
            return self.sentence_encoder(sentence1)


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
    model = SiameseNetwork(Config)
    s1 = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    s2 = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    l = torch.LongTensor([[1],[0]])
    y = model(s1, s2, l)
    print(y)
    # print(model.state_dict())