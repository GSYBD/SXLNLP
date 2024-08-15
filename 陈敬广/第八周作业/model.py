'''
建立网络模型结构，表示型 cos
'''
import torch
import torch.nn as nn
from torch.optim import Adam, SGD

'''
Encoder模型结构
'''


class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config['hidden_size']
        vocab_size = config['vocab_size'] + 1
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True,bidirectional=True)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, sentence):
        x = self.embedding(sentence)
        # 使用lstm
        # x, _ = self.lstm(x)
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        # self.loss = nn.TripletMarginLoss()

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        # dim=-1 最后一维归一化
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        # torch.mul用于执行张量之间的逐元素乘法
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin
        # 计算张量diff其中大于0的元素的平均值
        return torch.mean(diff[diff.gt(0)])

    def forward(self, sentence1, sentence2=None, sentence3=None):
        # 同时传入两个句子
        if sentence2 is not None:
            vector1 = self.sentence_encoder(sentence1)
            vector2 = self.sentence_encoder(sentence2)
            # 如果有标签，则计算loss
            if sentence3 is not None:
                vector3 = self.sentence_encoder(sentence3)
                # return self.loss(vector1, vector2, vector3)
                return self.cosine_triplet_loss(vector1, vector2, vector3)
            else:
                # 如果无标签，计算余弦距离
                return self.cosine_distance(vector1, vector2)
        else:
            # 单独传入一个句子时，认为正在使用向量化能力，实际预测的时候
            return self.sentence_encoder(sentence1)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    from config import Config

    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    l = torch.LongTensor([[1], [0]])
    y = model(s1, s2)
    print(y)
    # print(model.state_dict())
