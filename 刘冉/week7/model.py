# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertTokenizer, BertModel

'''
网络模型结构
'''

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        class_num = config["class_num"]
        model_type = config["model_type"]
        if "bert" in model_type:
            self.use_bert = True
            self.is_bert = (model_type == "bert")
            self.encoder, hidden_size = self.getBertEncoder(config, model_type)
        else:
            self.use_bert = False
            hidden_size = config["hidden_size"]
            vocab_size = config["vocab_size"] + 1
            self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
            if model_type == "fast_text":
                self.encoder = lambda x: x
            elif model_type == "lstm":
                self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=config["num_layers"])
            elif model_type == "gru":
                self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=config["num_layers"])
            elif model_type == "rnn":
                self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=config["num_layers"])
            elif model_type == "cnn":
                self.encoder = CNN(config)
            elif model_type == "gated_cnn":
                self.encoder = GatedCNN(config)
            elif model_type == "stack_gated_cnn":
                self.encoder = StackGatedCNN(config)
            elif model_type == "rcnn":
                self.encoder = RCNN(config)

        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.functional.cross_entropy  #loss采用交叉熵损失

    def forward(self, x, y=None):
        if self.use_bert:
            x = self.encoder(x)
        else:
            x = self.embedding(x) #
            x = self.encoder(x)
        if isinstance(x, tuple): #RNN类的模型会同时返回隐单元向量，我们只取序列结果
            x = x[0]
        elif self.is_bert:
            x = x[0]

        #采用pooling的方式得到句向量
        if self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1, 2)).squeeze()

        y_pre = self.classify(x)
        if y is not None:
            return self.loss(y_pre, y.squeeze())
        else:
            return y_pre

    def getBertEncoder(self, config, bert_type):
        if bert_type == "bert":
            bret_model = BertModel.from_pretrained(config["pretrain_model_path"])
            hidden_size = bret_model.config.hidden_size
        elif bert_type == "bert_lstm":
            bret_model = BertLSTM(config)
            hidden_size = bret_model.bert.config.hidden_size
        elif bert_type == "bert_cnn":
            bret_model = BertCNN(config)
            hidden_size = bret_model.bert.config.hidden_size
        elif bert_type == "bert_mid_layer":
            bret_model = BertMidLayer(config)
            hidden_size = bret_model.bert.config.hidden_size
        return bret_model, hidden_size


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1)/2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x): #x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)

class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)

    def forward(self, x):
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b)

class StackGatedCNN(nn.Module):
    def __init__(self, config):
        super(StackGatedCNN, self).__init__()
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        #ModuleList类内可以放置多个模型，取用时类似于一个列表
        self.gcnn_layers = nn.ModuleList(
            GatedCNN(config) for i in range(self.num_layers)
        )
        self.ff_liner_layers1 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.ff_liner_layers2 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_gcnn = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_ff = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )

    def forward(self, x):
        #仿照bert的transformer模型结构，将self-attention替换为gcnn
        for i in range(self.num_layers):
            gcnn_x = self.gcnn_layers[i](x)
            x = gcnn_x + x  #通过gcnn+残差
            x = self.bn_after_gcnn[i](x)  #之后bn
            # # 仿照feed-forward层，使用两个线性层
            l1 = self.ff_liner_layers1[i](x)  #一层线性
            l1 = torch.relu(l1)               #在bert中这里是gelu
            l2 = self.ff_liner_layers2[i](l1) #二层线性
            x = self.bn_after_ff[i](x + l2)        #残差后过bn
        return x

class RCNN(nn.Module):
    def __init__(self, config):
        super(RCNN, self).__init__()
        hidden_size = config["hidden_size"]
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.cnn = GatedCNN(config)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.cnn(x)
        return x

class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"])
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]
        x, _ = self.rnn(x)
        return x

class BertCNN(nn.Module):
    def __init__(self, config):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"])
        config["hidden_size"] = self.bert.config.hidden_size
        self.cnn = CNN(config)

    def forward(self, x):
        x = self.bert(x)[0]
        x = self.cnn(x)
        return x

class BertMidLayer(nn.Module):
    def __init__(self, config):
        super(BertMidLayer, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"])
        self.bert.config.output_hidden_states = True

    def forward(self, x):
        layer_states = self.bert(x)[2]
        layer_states = torch.add(layer_states[-2], layer_states[-1])
        return layer_states

#优化器选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)