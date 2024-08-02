"""
创建网络模型结构
"""

import torch.nn as nn
from transformers import BertModel
from config import config
from torch.optim import Adam, SGD

class Module(nn.Module):
    def __init__(self, config):
        super(Module, self).__init__()
        vocab = config["vocab_path"]
        hidden_size = config["hidden_size"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        pretrain_model_path = config["pretrain_model_path"]
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]
        self.bert = False
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if model_type == "fast_text":
            self.encoder = lambda x: x
        if model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        if model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        if model_type == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        if model_type == "bert":
            self.bert = True
            self.encoder = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        if model_type == "bert_lstm":
            self.bert = True
            self.encoder = Bert_LSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size

        self.linear = nn.Linear(hidden_size, class_num)
        self.pooling = config["pooling_style"]
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        if self.bert:
            x = self.encoder(x)
        else:
            x = self.embedding(x)
            x = self.encoder(x)

        if isinstance(x, tuple):
            x = x[0]

        if self.pooling == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1, 2)).squeeze()

        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y.squeeze())
        else:
            return y_pred

def choose_optimizer(config, model):
    optimizer = config["optim"]
    learning_rate = config["lr"]
    if optimizer == "Adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)

class Bert_LSTM(nn.Module):
    def __init__(self, config):
        super(Bert_LSTM, self).__init__()
        hidden_size = config["hidden_size"]
        self.encoder_bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.encoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=config["num_layers"], batch_first=True)

    def forward(self, x):
        x = self.encoder_bert(x)[0]
        x, _ = self.encoder_lstm(x)
        return x
