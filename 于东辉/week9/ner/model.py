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
        # 由于使用BERT，自定义的embedding层不再需要
        self.embedding = None

        # 加载BERT模型，同时获取BERT的配置信息
        self.bert = BertModel.from_pretrained(config["bert_path"])

        # 使用BERT模型的hidden_size来定义下游任务的网络层
        hidden_size = self.bert.config.hidden_size

        # 从BERT模型配置中获取vocab_size
        vocab_size = self.bert.config.vocab_size

        # 其他配置信息
        max_length = config.get("max_length")  # 假设默认最大长度为128
        class_num = config["class_num"]

        # 定义分类层，将BERT的输出映射到指定数量的类别上
        self.classify = nn.Linear(hidden_size, class_num)

        # CRF层在此配置中不使用
        self.use_crf = False

        # 定义损失函数，使用交叉熵损失，忽略-1索引的值
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, target=None):
        # 通过BERT模型获取序列输出
        outputs = self.bert(x, return_dict=False)
        sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)

        # 使用分类层获取预测结果，这里假设 num_tags 是类别数
        predict = self.classify(sequence_output)  # (batch_size, seq_len, num_tags)

        print(predict.shape)
        print(predict.view(-1, predict.shape[-1]).shape)
        if target is not None:
            print(target.shape)
            print(target.view(-1).shape)

        # 如果提供了目标标签，计算损失；否则，只返回预测结果
        if target is not None:
            # 确保 predict 和 target 形状匹配
            loss = self.loss(predict.contiguous().view(-1, predict.shape[-1]), target.contiguous().view(-1))
            return loss
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
