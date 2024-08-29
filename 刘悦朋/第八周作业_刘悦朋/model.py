import torch.nn as nn
import torch.nn.functional
from torch.optim import Adam, SGD
from config import Config

"""

    建立网络模型结构

"""


class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config['hidden_size']
        vocab_size = config['vocab_size'] + 1
        max_length = config['max_length']
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, bidirectional=False)
        self.layer = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        # self.dropout = nn.Dropout(0.5)

    # 输入为问题字符编码
    def forward(self, x):
        x = self.embedding(x)  # (batch_size, length) -> (batch_size, length, embedding_dim)
        # 使用lstm, x是序列结果
        x, _ = self.lstm(x)  # (batch_size, length, input_size) -> (batch_size, length, hidden_size)
        # 使用线性层
        # x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = nn.CosineEmbeddingLoss()

    # 计算余弦距离 1-cos(a, b)
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        return torch.mean(diff[diff.gt(0)])  # greater than

    # sentence: (batch_size, max_length)
    def forward(self, a, p=None, n=None):
        if p is not None:
            vector1 = self.sentence_encoder(a)
            vector2 = self.sentence_encoder(p)
            vector3 = self.sentence_encoder(n)
            return self.cosine_triplet_loss(vector1, vector2, vector3)
        else:
            return self.sentence_encoder(a)


def choose_optimizer(config, model):
    optimizer = config['optimizer']
    learning_rate = config['learning_rate']
    if optimizer == 'adam':
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    Config['vocab_size'] = 10
    Config['max_length'] = 4
    model = SiameseNetwork(Config)
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s3 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[1, 9, 9, 7], [0, 6, 0, 8]])
    y = model(s1, s2, s3)
    print(y)
    # print(model.state_dict())
