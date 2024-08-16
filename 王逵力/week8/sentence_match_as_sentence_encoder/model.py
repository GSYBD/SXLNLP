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
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    #输入为问题字符编码
    def forward(self, x):
        x = self.embedding(x)
        #使用lstm
        # x, _ = self.lstm(x)
        #使用线性层
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = nn.TripletMarginLoss(margin=0.1)
        self.config = config

    def cosine_distance(self, vector1, vector2):
        return 1.0 - self.cosine_similarity(vector1, vector2).mean()

    def forward(self, anchor, positive, negative=None, is_training=False):
        if is_training:
            # 在训练模式下，我们期望传入三个句子
            assert positive is not None and negative is not None, "Training requires all three sentences (anchor, positive, negative)"

            vector_anchor = self.sentence_encoder(anchor)  # vec: (batch_size, hidden_size)
            vector_positive = self.sentence_encoder(positive)
            vector_negative = self.sentence_encoder(negative)

            loss = self.loss(vector_anchor, vector_positive, vector_negative)
            return loss
        else:
            # 如果只有一个句子传入，我们假设是在使用向量化能力
            if positive is None and negative is None:
                return self.sentence_encoder(anchor)

            # 如果有两个句子传入，但不是训练模式，我们返回它们之间的余弦距离
            elif negative is None:
                vector_anchor = self.sentence_encoder(anchor)
                vector_positive = self.sentence_encoder(positive)
                distance = self.cosine_distance(vector_anchor, vector_positive)
                return distance

            # 如果三个句子都传入，但在非训练模式下，我们假设是想获取三者的向量
            else:
                vector_anchor = self.sentence_encoder(anchor)
                vector_positive = self.sentence_encoder(positive)
                vector_negative = self.sentence_encoder(negative)
                return vector_anchor, vector_positive, vector_negative


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