# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
# from torchcrf import CRF
from transformers import BertModel

"""
建立网络模型结构
使用bert
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        class_num = config["class_num"]

        self.encoder = BertModel.from_pretrained(config['bert_path'], return_dict=False)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        hidden_size = self.encoder.config.hidden_size
        self.classify = nn.Linear(hidden_size, class_num)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # loss采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, attention_mask=None, target=None):
        """
        x: 8句话，50个字，756维
        [8, 50, 768]

        predict [8, 50, 9]
        :param x:
        :param target:
        :return:
        """

        x,_= self.encoder(x, attention_mask=attention_mask)  # input shape:(batch_size, sen_len)

        x = self.dropout(x)
        predict = self.classify1(x)  # ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)
        predict = self.classify2(predict)  # ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if target is not None:
            return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            return predict


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
