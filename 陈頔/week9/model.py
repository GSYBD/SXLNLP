# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        max_length = config["max_length"]
        class_num = config["class_num"]
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        # 使用bert
        self.bert = BertModel.from_pretrained(config["bert_path"], return_dict=False)
        self.classify = nn.Linear(self.bert.config.hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        # x = self.embedding(x)  #input shape:(batch_size, sen_len)
        # x, _ = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)
        # lstm不用做pooling，可以直接进线性层

        x, _ = self.bert(x)
        #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)
        #因为交叉熵损失函数要求输入的形状为(batch_size, num_classes)，所以要把前2维乘起来
        predict = self.classify(x) 

        if target is not None:
            if self.use_crf:
                # 因为loader中把多余的填成-1，所以这里大于-1
                mask = target.gt(-1)
                # crf调用的是crf的forward的函数，会算出分数llh.sum()/mask.float().sum,
                # llh = numerator - denominator ，要把这个颠倒过来才能作为loss，所以加负号，也替代了原来的loss（是否加负号跟库有关系）
                # 如果这个作为loss，就替换了交叉熵做loss
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                #(number, class_num), (number)
                # -1表示展平，打通数据内部的维度，将所有数据展平
                # predict.view(-1, predict.shape[-1])将predict张量重塑成一个二维张量
                # 其中第二维的大小保持不变（即最后一个维度的大小）而第一维的大小则根据元素总数自动计算
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                # 如果用crf，会涉及到一个专门的decode解码方法，涉及到转移矩阵和发射矩阵怎么算路径的分数，涉及到篱笆墙解码
                return self.crf_layer.decode(predict)
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