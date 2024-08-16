# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertForTokenClassification
# from torchcrf import CRF
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    class TorchModel(nn.Module):
        def __init__(self, config):
            super(TorchModel, self).__init__()
            self.bert = BertForTokenClassification.from_pretrained(config["bert_path"], num_labels=config["class_num"])
            self.use_crf = config["use_crf"]
            self.crf_layer = CRF(config["class_num"], batch_first=True) if self.use_crf else None

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input_ids, attention_mask=None, target=None):
        # 获取 BERT 的输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch_size, sequence_length, num_labels)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)  # mask for CRF
                return - self.crf_layer(logits, target, mask, reduction="mean")
            else:
                loss = nn.CrossEntropyLoss(ignore_index=-1)
                return loss(logits.view(-1, logits.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(logits)
            else:
                return logits


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