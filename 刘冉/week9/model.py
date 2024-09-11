# coding: utf-8

import torch
import torch.nn as nn
from transformers import BertModel
from torch.optim import Adam, SGD
'''
模型
'''

class PunctuationModel(nn.Module):
    def __init__(self, config):
        super(PunctuationModel, self).__init__()
        class_num = config["class_num"]
        padding = config["padding"]
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"])
        hidden_size = self.bert.config.hidden_size
        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.classify = nn.Linear(hidden_size, class_num)
        self.loss = nn.CrossEntropyLoss(ignore_index=padding)

    def forward(self, x, target=None):
        x = self.bert(x)[0]
        # x, _ = self.lstm(x)
        predict = self.classify(x)
        if target is not None:
            predict = predict.view(-1, predict.shape[-1])
            target = target.view(-1)
            return self.loss(predict, target)
        else:
            return predict

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)