import torch.nn as nn
import torch.nn.functional
from transformers import BertModel
class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config['hidden_size']
        vocab_size = config['vocab_size'] + 1
        max_length = config['max_length']
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.use_bert = config['use_bert']
        if self.use_bert:
            self.encoder = BertModel.from_pretrained(config['pretrain_model_path'], return_dict=False)
        else:
            self.encoder = nn.LSTM(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.pooling_layer = nn.AvgPool1d(max_length)
    def forward(self,x):
        if self.use_bert:
            x = self.encoder(x)
        else:
            x = self.embedding(x)
            x = self.encoder(x)
        if isinstance(x, tuple):
            x = x[0]
        x = self.dropout(x)
        x = self.pooling_layer(x.transpose(1,2)).squeeze()
        return x

class SeeNetWork(nn.Module):
    def __init__(self, config):
        super(SeeNetWork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = nn.CosineEmbeddingLoss()

    def cosine_distance(self, t1, t2):
        t1 = torch.nn.functional.normalize(t1, dim=-1)
        t2 = torch.nn.functional.normalize(t2, dim=-1)
        cosine = torch.sum(torch.mul(t1, t2), dim=-1)
        return 1 - cosine
    def triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        return torch.mean(diff[diff.ge(0)])


    def forward(self, s1, s2=None, s3=None):
        if s2 is not None:
            v1 = self.sentence_encoder(s1)
            v2 = self.sentence_encoder(s2)
            v3 = self.sentence_encoder(s3)
            return self.triplet_loss(v1, v2, v3)
        else:
            return self.sentence_encoder(s1)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)