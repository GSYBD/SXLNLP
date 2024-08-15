# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import config
from transformers import BertModel

class TorchModel(nn.Module):
        def __init__(self,config):
            super(TorchModel,self).__init__()
            hidden_size = config['hidden_size']
            vocab_size = config["vocab_size"] + 1
            class_num = config["class_num"]
            model_type = config['model_type']
            num_layers = config["num_layers"]
            self.use_bert = False
            self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
            if model_type == 'lstm':
                self.encoder = nn.LSTM(hidden_size,hidden_size,num_layers=num_layers,batch_first=True)
            elif model_type == 'rnn':
                self.encoder = nn.RNN(hidden_size,hidden_size,num_layers=num_layers,batch_first=True)
            elif model_type == 'gated_cnn':
                self.encoder = GateCNN(config)
            elif model_type == 'bert':
                self.use_bert = True
                self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
                hidden_size = self.encoder.config.hidden_size
            elif model_type == 'bert_cnn':
                self.use_bert = True
                self.encoder = BertCNN(config)
                hidden_size = self.encoder.bert.config.hidden_size

            self.classify = nn.Linear(hidden_size,class_num)
            self.pooling_style = config["pooling_style"]
            self.loss = nn.functional.cross_entropy
        def forward(self,x,target = None):
            if self.use_bert:
                x = self.encoder(x)
            else:
                x =self.embedding(x)

            if isinstance(x,tuple):
                x =x[0]

            if self.pooling_style == 'max':
                self.pooling_layer = nn.MaxPool1d(x.shape[1])  # 最大池化
            else:
                self.pooling_layer = nn.AvgPool1d(x.shape[1])
            x = self.pooling_layer(x.transpose(1,2)).squeeze()

            predict = self.classify(x)
            if target is not None:
                return  self.loss(predict,target.squeeze())
            else:
                return predict






"""
这个模型类实现了一个简单的一维卷积神经网络，通过对输入数据进行一维卷积操作，可以用于处理序列数据，如文本分类、序列建模等任务。
"""
class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]  # 从配置中获取隐藏层大小 hidden_size。
        kernel_size = config["kernel_size"]  # 从配置中获取卷积核大小 kernel_size。
        pad = int((kernel_size - 1)/2)  # 计算填充大小，使得卷积操作的输出大小与输入大小相同
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)  # 定义一个一维卷积层 nn.Conv1d，包括输入通道数、输出通道数、卷积核大小、是否包含偏置项和填充大小

    def forward(self, x): #x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)  #对输入进行一维卷积操作，首先通过 transpose(1, 2) 调整维度，然后将卷积结果再次通过 transpose(1, 2) 调整维度，最终返回卷积层的输出。

""" 
做两个卷积   一个标注卷积和一个过激活函数的卷积
"""
class GateCNN(nn.Module):
    def __init__(self, config):
        super(GateCNN, self).__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)

    def forward(self, x):
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b)  # 将 a 和 b 逐元素相乘，实现门控机制的效果，得到模型的最终输出。

class BertCNN(nn.Module):
    def __init__(self, config):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        config["hidden_size"] = self.bert.config.hidden_size
        self.cnn = CNN(config)

    def forward(self, x):
        x = self.bert(x)[0]
        x = self.cnn(x)
        return x
from torch.optim import Adam, SGD

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(),lr=learning_rate)

if __name__ == '__main__':
    from config import  Config

    Config["model_type"] = 'bert'
    model = BertModel.from_pretrained(Config["pretrain_model_path"], return_dict=False)
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    sequence_output, pooler_output = model(x)
    print("x[1]:",x[1],"type(x[1]):", type(x[1]), "len(x[1]):",len(x[1]))

    # model = TorchModel(Config)
    # label = torch.LongTensor([1,2])
    # print(model(x, label))