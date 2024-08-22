# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, BertTokenizer
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
    #     self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
    #     self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
    #     self.classify = nn.Linear(hidden_size * 2, class_num)
    #     self.crf_layer = CRF(class_num, batch_first=True)
    #     self.use_crf = config["use_crf"]
    #     self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    # #当输入真实标签，返回loss值；无真实标签，返回预测值
    # def forward(self, x, target=None):
    #     x = self.embedding(x)  #input shape:(batch_size, sen_len)
    #     x, _ = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)
    #     predict = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

    #     if target is not None:
    #         if self.use_crf:
    #             mask = target.gt(-1) 
    #             return - self.crf_layer(predict, target, mask, reduction="mean")
    #         else:
    #             #(number, class_num), (number)
    #             return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
    #     else:
    #         if self.use_crf:
    #             return self.crf_layer.decode(predict)
    #         else:
    #             return predict
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.bert_model_name = config["bert_model_name"]
        self.bert = BertModel.from_pretrained(self.bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        hidden_size = self.bert.config.hidden_size
        class_num = config["class_num"]
        self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # Loss采用交叉熵损失

    def forward(self, input_id, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(input_id, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        logits = self.classify(sequence_output)  # (batch_size, seq_len, num_tags)

        if labels is not None:
            if self.use_crf:
                mask = labels.gt(-1)
                # The CRF layer should output a scalar loss if reduction="mean" is used
                loss = -self.crf_layer(logits, labels, mask, reduction="mean")
            else:
                # CrossEntropyLoss reduction should be mean or sum
                loss = self.loss(logits.view(-1, logits.shape[-1]), labels.view(-1))
            # Confirm loss is scalar
            # print(f"Loss is scalar: {loss.item()}")
            return loss
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