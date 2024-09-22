# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.optim import Adam, SGD
from torchcrf import CRF

"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        class_num = config["class_num"]
        self.use_bert = config["use_bert"]
        if self.use_bert:
            self.bert_layer = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            hidden_size = self.bert_layer.config.hidden_size
            # self.bert_layer.encoder.layer = config.get("bert_layer_nums", 2) 行不通，直接改本地文件config吧
            self.bert_classify = nn.Linear(hidden_size, class_num)
        else:
            hidden_size = config["hidden_size"]
            vocab_size = config["vocab_size"] + 1
            self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
            self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=1)
            self.classify = nn.Linear(hidden_size * 2, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # loss采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, mask, target=None):
        if self.use_bert:
            bert_output = self.bert_layer(input_ids=x, attention_mask=mask)  # 包含了embedding和transformer共同作为encoder
            if isinstance(bert_output, tuple):
                sequence_output = bert_output[0]
            else:
                sequence_output = bert_output.last_hidden_state  # shape: (batch_size, seq_length, hidden_size)
            predict = self.bert_classify(sequence_output)  # shape: (batch_size, seq_length, class_num)
        else:
            x = self.embedding(x)   # input shape:(batch_size, sen_len)
            x, _ = self.layer(x)      # input shape:(batch_size, sen_len, input_dim) input_dim == hidden_size
            predict = self.classify(x)
        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.viterbi_decode(predict)
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

# # 数据流转梳理：load读取训练数据，使用enumerate枚举取出 第一次过model进行训练 计算loss并打印
# # 每个epoch结束调用测试类读取valid数据，调用eval函数，内部再次load，枚举，过model计算acc
# # predict函数使用。模型部署；
# # 其中predict和eval涉及模型输出，需要分类处理
