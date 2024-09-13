# coding: utf-8

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.optim import Adam, SGD
'''
模型
'''

class SFTModel(nn.Module):
    def __init__(self, config):
        super(SFTModel, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.bertTokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        hidden_size = self.bert.config.hidden_size
        self.classify = nn.Linear(hidden_size, self.bertTokenizer.vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.bert.config.pad_token_id)

    def forward(self, x, y=None, mask=None):
        if mask is not None:
            x, _ = self.bert(x, attention_mask=mask)
        else:
            x, _ = self.bert(x)
        predict = self.classify(x)
        if y is not None:
            predict = predict.view(-1, predict.shape[-1])
            target = y.view(-1)
            return self.loss(predict, target)
        else:
            return torch.softmax(predict, dim=-1)

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)