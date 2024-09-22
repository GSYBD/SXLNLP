import torch
import torch.nn as nn
from transformers import BertModel
from config import Config


class STFModel(nn.Module):
    def __init__(self):
        super(STFModel, self).__init__()
        self.bert = BertModel.from_pretrained(Config['bert'], return_dict=False)
        self.classify = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, mask=None, y=None):
        if y is not None:
            x, _ = self.bert(x, attention_mask=mask)
            y_predict = self.classify(x)
            return self.loss(y_predict.view(-1, y_predict.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_predict = self.classify(x)
            return torch.softmax(y_predict, dim=-1)
