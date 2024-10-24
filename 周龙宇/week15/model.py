# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import T5Model
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=1)
        self.bio_classifier = nn.Linear(hidden_size * 2, config["bio_count"])
        self.attribute_classifier = nn.Linear(hidden_size * 2, config["attribute_count"])
        self.attribute_loss_ratio = config["attribute_loss_ratio"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, attribute_target=None, bio_target=None):
        x = self.embedding(x)
        x, _ = self.layer(x)  #(batch_size, max_length, hidden_size)
        #序列标注
        bio_predict = self.bio_classifier(x)  #(batch_size, max_length, 5) (Head-B, Head-I, Tail-B, Tail-I, O)
        #文本分类
        self.pooling_layer = nn.AvgPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1, 2)).squeeze() #inpt:(batch, sen_len, hidden_size) -> (batch, hidden_size)
        attribute_predict = self.attribute_classifier(x) #(batch, hidden_size) ->(batch, class)
        #multi-task训练
        if bio_target is not None:
            bio_loss = self.loss(bio_predict.view(-1, bio_predict.shape[-1]), bio_target.view(-1))
            attribute_loss = self.loss(attribute_predict.view(x.shape[0], -1), attribute_target.view(-1))
            return bio_loss + attribute_loss * self.attribute_loss_ratio
        else:
            return attribute_predict, bio_predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)