import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel


class SftModel(nn.Module):
    def __init__(self, config):
        super(SftModel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config['bert_path'], return_dict=False)
        self.classifier = nn.Linear(config['hidden_size'], config['vocab_size'])
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x, y=None, mask=None):
        if y is not None:
            x, _ = self.bert(x, attention_mask=mask)  # x.shape(batch_size,max_len,hidden_size)
            y_pred = self.classifier(x)  # y_pred.shape(batch_size,max_len,config['vocab_size'])
            return self.loss(y_pred.view(-1,y_pred.shape[-1]),y.view(-1))
        else:
            x, _ = self.bert(x)  # x.shape(batch_size,max_len,hidden_size)
            y_pred = self.classifier(x)
            return torch.softmax(y_pred, dim=-1)  # 对最后一维softmax