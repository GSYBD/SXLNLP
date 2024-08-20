import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from create_datas  import *
from   config  import config

class Sensents_Encoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        vocab_path = config["vocab_path"]
        with open(vocab_path, 'r', encoding='utf8') as f:
            vocab_size = len(f.readlines())
        hidden_size = config["hidden_size"]
        self.embedding = nn.Embedding(vocab_size+1,hidden_size)
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.pool = nn.MaxPool1d(config['sentence_len'])
    def forward(self,x):
        x = self.embedding(x)
        x = self.layer1(x)
        x = self.pool(x.transpose(1, 2)).squeeze()
        return x


class SiameseNetwork(nn.Module):
    """ 这个函数实现了  1（sentens1，sentens2,target）(target是事先判断好的相近的就认为是1，反之0)
                     2（sentens1） 为了测试集准备的 计算句子的向量
                     3 (anchor,positive,negative) 这种方式来训练
    """
    def __init__(self,config):
        super().__init__()
        self.encoder = Sensents_Encoder(config)
        self.loss = nn.CosineEmbeddingLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=config['margin'], p=2)
    # def cosine_distance(self,tensor1, tensor2):
    #     tensor1 = torch.nn.functional.normalize(tensor1,dim=-1)
    #     tensor2 = torch.nn.functional.normalize(tensor1,dim=-1)
    #     cosine = torch.sum(torch.mul(tensor1,tensor2),axis=-1)
    #     return 1-cosine
    # def triplet_forward(self,anchor,positive,negative):
    #     loss = self.triplet_loss(anchor,positive,negative)
    #     return loss
    def forward(self,sentence1,sentence2=None,sentence3=None,target=None):
        if sentence2 is  None:
            tensor1 = self.encoder(sentence1)
            return tensor1
        elif sentence3 is not None:
            tensor1 = self.encoder(sentence1)
            anchor = tensor1
            tensor2 = self.encoder(sentence2)
            positive = tensor2
            tensor3 = self.encoder(sentence3)
            negative = tensor3
            return self.triplet_loss(anchor,positive,negative)
        elif target is not  None:
            tensor1 = self.encoder(sentence1)
            tensor2 = self.encoder(sentence2)
            return self.loss(tensor1,tensor2,target)

def choose_optimizer(config,model):
    if config['optimizer'] == 'adam':
        return torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config['optimizer'] == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=config['learning_rate'])


if __name__ == '__main__':
    model = SiameseNetwork(config)
    s1 = torch.LongTensor([[1, 2, 3, 0,2,2,2], [2, 2, 0, 0,2,2,2]])
    s2 = torch.LongTensor([[1, 2, 3, 4,2,2,2], [3, 2, 3, 4,2,2,2]])
    s3 = torch.LongTensor([[1, 2, 3, 4,2,2,2], [3, 2, 3, 4,2,2,2]])

    l = torch.LongTensor([[1], [0]]).squeeze()
    y = model(sentence1=s1, sentence2=s2,sentence3=s3)
    print(y)

