# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    # def __init__(self, config):
    #     super(TorchModel, self).__init__()
    #     hidden_size = config["hidden_size"]
    #     vocab_size = config["vocab_size"] + 1
    #     max_length = config["max_length"]
    #     class_num = config["class_num"]
    #     num_layers = config["num_layers"]
    #
    #     self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
    #     self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
    #     self.classify = nn.Linear(hidden_size * 2, class_num)
    #     self.crf_layer = CRF(class_num, batch_first=True) # lyu: 主要作用?
    #     self.use_crf = config["use_crf"]
    #     self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失
    #
    # #当输入真实标签，返回loss值；无真实标签，返回预测值
    # def forward(self, x, target=None):
    #     x = self.embedding(x)  #input shape:(batch_size, sen_len)
    #     x, _ = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)  # lyu: 不用做pooling, 也不用取最后一个元素, 直接用LSTM的输出
    #     predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)
    #
    #     if target is not None:  # lyu: 训练过程有真实值，则计算loss
    #         if self.use_crf:
    #             mask = target.gt(-1)  # lyu: mask的作用是过滤掉padding的部分, crf底层会根据mask来计算loss, -1表示padding(在loader中已经处理过))
    #             return - self.crf_layer(predict, target, mask, reduction="mean") # lyu: 添加负号是因为当前库的CRF层默认是取对数概率(最大化)，而交叉熵损失函数需要的是真实概率(最小化)，所以需要取负号; 底层调用的 是forward()函数; 可以理解为替换了交叉熵作为loss
    #         else:
    #             #(number, class_num), (number) # lyu: 如果有真实值，则不使用crf, 直接返回loss值
    #             return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1)) # lyu: 该方式与分词写法一致
    #     else:  # lyu: 预测过程无真实值，则返回预测值
    #         if self.use_crf:  # lyu: 如果使用crf, 则采用专门的方法,  返回crf的预测值
    #             return self.crf_layer.decode(predict)
    #         else:
    #             return predict  # lyu: 如果不使用crf. 正常输出每个字对应的向量, 取对应向量最大的那一维度作为预测值

    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]

        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        # 使用bert模型
        self.encoder = BertModel.from_pretrained(config["bert_path"], return_dict=False)
        hidden_size = self.encoder.config.hidden_size

        # self.classify = nn.Linear(hidden_size * 2, class_num)
        self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # loss采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        # x = self.embedding(x)  #input shape:(batch_size, sen_len)
        # x, _ = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)  # lyu: 不用做pooling, 也不用取最后一个元素, 直接用LSTM的输出
        x, _ = self.encoder(x)  # lyu: bert自带embdding, 如果模型使用bert, 参数直接传入模型即可, 其他模型需要先传入embdding层

        predict = self.classify(x)  # ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if target is not None:  # lyu: 训练过程有真实值，则计算loss
            if self.use_crf:
                mask = target.gt(-1)  # lyu: mask的作用是过滤掉padding的部分, crf底层会根据mask来计算loss, -1表示padding(在loader中已经处理过))
                return - self.crf_layer(predict, target, mask,
                                        reduction="mean")  # lyu: 添加负号是因为当前库的CRF层默认是取对数概率(最大化)，而交叉熵损失函数需要的是真实概率(最小化)，所以需要取负号; 底层调用的 是forward()函数; 可以理解为替换了交叉熵作为loss
            else:
                # (number, class_num), (number) # lyu: 如果有真实值，则不使用crf, 直接返回loss值
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))  # lyu: 该方式与分词写法一致
        else:  # lyu: 预测过程无真实值，则返回预测值
            if self.use_crf:  # lyu: 如果使用crf, 则采用专门的方法,  返回crf的预测值
                return self.crf_layer.decode(predict)
            else:
                return predict  # lyu: 如果不使用crf. 正常输出每个字对应的向量, 取对应向量最大的那一维度作为预测值


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