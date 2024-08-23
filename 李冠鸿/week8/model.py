import torch
import torch.nn as nn
import torch.nn.functional
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
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        sentence_length = torch.sum(x.gt(0), dim=-1)
        x = self.embedding(x)
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = nn.CosineEmbeddingLoss()

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
            diff = ap - an + margin.squeeze()
        return torch.mean(diff[diff.gt(0)])

    def forward(self, sentence1, sentence2=None, sentence3=None):
        if sentence2 is not None and sentence3 is not None:
            vector1 = self.sentence_encoder(sentence1)
            vector2 = self.sentence_encoder(sentence2)
            vector3 = self.sentence_encoder(sentence3)
            return self.cosine_triplet_loss(vector1, vector2, vector3)
        else:
            return self.sentence_encoder(sentence1)

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "Adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)

if __name__ == '__main__':
    from config import Config
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[1,2,3,4],[3,2,3,4]])
    l = torch.LongTensor([[1],[0]])
    y = model(s1, s2, 1)
    print(y)
