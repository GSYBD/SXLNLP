# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
"""
建立网络模型结构
lyu: 文本匹配--表示型示例
"""

# lyu: 定义模型, 输入一个文本，输出一个向量
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
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()  # lyu: 池化层:  max pooling
        return x   # lyu: (batch_size, hidden_size)

# lyu: 定义模型, 输入两个文本，输出一个向量
class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        # self.loss = nn.CosineEmbeddingLoss()  # lyu: 余弦距离损失函数
        # self.loss = nn.TripletMarginLoss(margin=0.1)  # lyu: 三元组损失函数

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1) # lyu: -1表示最后一个维度
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1) # lyu: 两个向量先进行归一化, 然后进行点积运算(对位相乘再相加), 按最后一个维度求和
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=None):  # lyu: 计算三元组损失函数, a与p相似，a与n不相似
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        return torch.mean(diff[diff.gt(0)]) #greater than   # lyu: 求大于0的元素的平均值, 是因为计算整个batch的平均损失

    # #sentence : (batch_size, max_length)
    # #lyu: 表示性文本匹配任务, encoding中输入两个句子，每个句子转换为一个向量; 然后经过loss计算，如果有标签,得到损失值,有3种可能的输出
    # def forward(self, sentence1, sentence2=None, target=None):
    #     #同时传入两个句子
    #     if sentence2 is not None:
    #         # lyu: 对应u, 区别与交互型文本匹配处理方式, 这里的一句话其实真的就是一个问题
    #         vector1 = self.sentence_encoder(sentence1)  # vec:(batch_size, hidden_size)
    #         vector2 = self.sentence_encoder(sentence2)  # lyu: 对应v
    #         #如果有标签，则计算loss
    #         if target is not None:
    #             return self.loss(vector1, vector2, target.squeeze()) # lyu: 传入真实的target,希望是正样本,还是负样本, 计算CosineEmbeddingLoss()
    #         #如果无标签，计算余弦距离
    #         else:
    #             return self.cosine_distance(vector1, vector2)
    #     #单独传入一个句子时，认为正在使用向量化能力
    #     else:
    #         return self.sentence_encoder(sentence1)

    def forward(self, sentence1, sentence2=None, sentence3=None):
        #同时传入3个句子,计算损失
        if sentence2 is not None and sentence3 is not None:
            vector1 = self.sentence_encoder(sentence1)  # vec:(batch_size, hidden_size)
            vector2 = self.sentence_encoder(sentence2)  # lyu:
            vector3 = self.sentence_encoder(sentence3)  # lyu:
            return self.cosine_triplet_loss(vector1, vector2, vector3)
        #单独传入一个句子时，认为正在使用向量化能力
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
    s3 = torch.LongTensor([[1,2,3,4], [1,2,3,4]])
    y = model(s1, s2, s3)
    # l = torch.LongTensor([[1],[0]])
    # y = model(s1, s2, l)
    print(y)
    # print(model.state_dict())