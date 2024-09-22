# -*- coding: utf-8 -*-
import torch
from torch import nn, Tensor

from transformers import BertTokenizer, BertModel


class XModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config["class_num"] = 3

        self.model_type = self.config["model_type"]

        if self.model_type != 'bert':
            self.embedding = nn.Embedding(self.config["vocab_size"], self.config["embedding_dim"], padding_idx=0)

        if self.model_type == 'linear':
            self.layer = nn.Linear(self.config["embedding_dim"], self.config["embedding_dim"])
        elif self.model_type == 'bert':
            self.bert = BertModel.from_pretrained(self.config["bert_model_path"], return_dict=False)
            self.tokenizer = BertTokenizer.from_pretrained(self.config["bert_model_path"])
            self.config["class_num"] = self.bert.config.vocab_size

        self.classify = nn.Linear(self.config["embedding_dim"], self.config["class_num"])
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None, mask=None):
        if y is not None:
            # 训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            # mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            # if torch.cuda.is_available():
            #     mask = mask.cuda()
            self.bert.attn_implementation = "xxx"
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            # 预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)  # output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)
