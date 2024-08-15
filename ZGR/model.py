# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF

"""
建立网络模型结构
"""

class BertCRF(nn.Module):
    def __init__(self, config):
        super(BertCRF, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"])
        self.classify = nn.Linear(config["hidden_size"], config["class_num"])
        self.crf_layer = CRF(config["class_num"], batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, attention_mask, target=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state
        predict = self.classify(x)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return -self.crf_layer(predict, target, mask, reduction="mean")
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict, attention_mask=attention_mask)
            else:
                return predict

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
