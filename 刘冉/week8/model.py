# coding: utf-8

import torch
import torch.nn as nn
from torch.optim import Adam, SGD

'''
模型结构
'''

class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1) / 2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        pooling_layer = nn.MaxPool1d(x.shape[1])
        x = pooling_layer(x.transpose(1, 2)).squeeze()
        return x


class TripletLossModel(nn.Module):
    def __init__(self, config):
        super(TripletLossModel, self).__init__()
        self.encoder = SentenceEncoder(config)
        self.margin = config["margin"]
        self.loss = nn.TripletMarginLoss(margin=self.margin)

    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin
        return torch.mean(diff[diff.gt(0)])

    def forward(self, sentence1, sentence2=None, sentence3=None):
        vector1 = self.encoder(sentence1)
        # 如果只有一个句子 则返回encoder
        if sentence2 is not None:
            # 两个句子 返回距离，三个则返回loss
            vector2 = self.encoder(sentence2)
            if sentence3 is not None:
                a = vector1
                p = vector2
                n = self.encoder(sentence3)
                return self.loss(a, p, n)
                #return self.cosine_triplet_loss(a, p, n, margin=self.margin)
            else:
                return self.cosine_distance(vector1, vector2)
        else:
            return vector1

def choose_optimizer(config, model):
    opt = config["optimizer"]
    learning_rate = config["learning_rate"]
    if opt == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif opt == "sgd":
        return SGD(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = TripletLossModel(Config)
    s1 = torch.LongTensor([[1,2,3,0], [2,2,0,0]])
    s2 = torch.LongTensor([[1,2,3,3], [2,2,0,0]])
    s3 = torch.LongTensor([[1,2,3,4], [3,2,3,4]])
    y = model(s1, s2, s3)
    print(y)
    # print(model.state_dict())