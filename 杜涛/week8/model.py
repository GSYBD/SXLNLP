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
        vocab_size = config["vocab_size"] + 1  # 预留pad的长度
        max_length = config["max_length"]
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


class SiameseNetwork(nn.Module):
    def __init__(self, config, margin=1.0):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)  # 向量化类用于处理question
        if config["loss_type"] == "cosine":  # 损失函数
            self.loss = nn.CosineEmbeddingLoss()
        else:
            self.loss = nn.TripletMarginLoss(margin=margin)

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)  # 元素乘法mul/multiply 矩阵乘法numpy.dot()/torch.matmul()
        return 1 - cosine  # 算法：A·B/|A|*|B| 为向量cos夹角 转化为 A,B标准化后做元素乘法 1为完全相似,-1为完全相反
        # 此处转化为值域在 [0，2]之间的 余弦距离 值为0时完全相似，值为2时完全相反。正交/垂直时为1

    def cosine_triplet_loss(self, a, p, n, margin=None):  # 三元组损失函数
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        return torch.mean(diff[diff.gt(0)]) #greater than

    def forward(self, sentence1, sentence2=None, sentence3=None):
        vec1 = self.sentence_encoder(sentence1)
        if sentence2 is not None:
            vec2 = self.sentence_encoder(sentence2)
            if sentence3 is not None:
                vec3 = self.sentence_encoder(sentence3)
                return self.loss(vec1, vec2, vec3)
            return self.cosine_distance(vec1, vec2)
        else:
            return vec1

    # # sentence : (batch_size, max_length)
    # def forward(self, sentence1, sentence2=None, target=None):
    #     # 同时传入两个句子
    #     if sentence2 is not None:
    #         vector1 = self.sentence_encoder(sentence1) #vec:(batch_size, hidden_size)
    #         vector2 = self.sentence_encoder(sentence2)
    #         # 如果有标签，则计算loss
    #         if target is not None:
    #             return self.loss(vector1, vector2, target.squeeze())
    #         # 如果无标签，计算余弦距离
    #         else:
    #             return self.cosine_distance(vector1, vector2)
    #     # 单独传入一个句子时，认为正在使用向量化能力
    #     else:
    #         return self.sentence_encoder(sentence1)


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