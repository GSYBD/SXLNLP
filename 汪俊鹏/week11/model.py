#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel
from config import Config
from loader import load_vocab
from loader import load_mask
"""
"""
import logging
logger = logging.getLogger(__name__)

class LanguageModel(nn.Module):
    def __init__(self, config):
        super(LanguageModel, self).__init__()
        # self.embedding = nn.Embedding(len(vocab), input_dim)
        # self.layer = nn.LSTM(input_dim, input_dim, num_layers=1, batch_first=True)
        self.config=Config
        self.bert = BertModel.from_pretrained(config['bert_path'], return_dict=False)
        self.vocab=load_vocab(config['vocab_path'])
        self.vocab_size=len(self.vocab)
        self.classify = nn.Linear(768, self.vocab_size)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, config,x, y=None):
        if y is not None:
            #训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            mask=logger(config["train_data_path"], config, logger)
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x,attention_mask=mask)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1),ignore_index=-99)
        else:
            #预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)





