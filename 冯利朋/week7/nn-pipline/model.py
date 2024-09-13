import torch.nn as nn
import torch.nn.functional
from torch.optim import Adam,SGD
from transformers import BertModel
class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.user_bert = config['use_bert']
        hidden_size = config['hidden_size']
        vocab_size = config['vocab_size'] + 1
        class_num = config['class_num']
        num_layers = config['num_layers']
        model_type = config['model_type']
        max_length = config['max_length']
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        if model_type == 'fast_text':
            self.encoder = lambda x: x
        elif model_type == 'rnn':
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type =='lstm':
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == 'gru':
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == 'cnn':
            self.encoder = CNN(config)
        elif model_type == 'gated_cnn':
            self.encoder = GatedCNN(config)
        elif model_type == 'stack_gated_cnn':
            self.encoder = StackGatedCNN(config)
        elif model_type == 'rcnn':
            self.encoder = RCNN(config)
        elif model_type == "bert":
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        elif model_type == "bert_lstm":
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_cnn":
            self.encoder = BertCNN(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_mid_layer":
            self.encoder = BertMidLayer(config)
            hidden_size = self.encoder.bert.config.hidden_size
        if config['pooling_style'] == 'max':
            self.pooling_layer = nn.MaxPool1d(max_length)
        else:
            self.pooling_layer = nn.AvgPool1d(max_length)
        self.classify = nn.Linear(hidden_size, class_num)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy
    def forward(self, x, y=None):
        if self.user_bert:
            x = self.encoder(x)[0]
        else:
            x = self.embedding(x)
            x = self.encoder(x)
        if isinstance(x, tuple):
            x = x[0]
        x = self.dropout(x)
        x = self.pooling_layer(x.transpose(1, 2)).squeeze()
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred, y.squeeze())
        else:
            return y_pred
class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1)/2)
        # hidden_size 输入维度
        # hidden_size 几个卷子核
        # kernel_size 核的大小
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x): #x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)
class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn = CNN(config)
        self.gated = CNN(config)
    def forward(self, x):
        x = self.cnn(x)
        y = self.gated(x)
        y = torch.sigmoid(y)
        return torch.mul(x, y)

class StackGatedCNN(nn.Module):
    def __init__(self, config):
        super(StackGatedCNN, self).__init__()
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        #ModuleList类内可以放置多个模型，取用时类似于一个列表
        self.gcnn_layers = nn.ModuleList(
            GatedCNN(config) for i in range(self.num_layers)
        )
        self.ff_liner_layers1 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.ff_liner_layers2 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_gcnn = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_ff = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )

    def forward(self, x):
        #仿照bert的transformer模型结构，将self-attention替换为gcnn
        for i in range(self.num_layers):
            gcnn_x = self.gcnn_layers[i](x)
            x = gcnn_x + x  #通过gcnn+残差
            x = self.bn_after_gcnn[i](x)  #之后bn
            # # 仿照feed-forward层，使用两个线性层
            l1 = self.ff_liner_layers1[i](x)  #一层线性
            l1 = torch.relu(l1)               #在bert中这里是gelu
            l2 = self.ff_liner_layers2[i](l1) #二层线性
            x = self.bn_after_ff[i](x + l2)        #残差后过bn
        return x
class RCNN(nn.Module):
    def __init__(self, config):
        super(RCNN, self).__init__()
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.cnn = CNN(config)
    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.cnn(x)
        return x
class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"],return_dict=False)
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]
        x, _ = self.rnn(x)
        return x

class BertCNN(nn.Module):
    def __init__(self, config):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"],return_dict=False)
        config["hidden_size"] = self.bert.config.hidden_size
        self.cnn = CNN(config)

    def forward(self, x):
        x = self.bert(x)[0]
        x = self.cnn(x)
        return x

class BertMidLayer(nn.Module):
    def __init__(self, config):
        super(BertMidLayer, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"],return_dict=False)
        self.bert.config.output_hidden_states = True

    def forward(self, x):
        layer_states = self.bert(x)[2]
        layer_states = torch.add(layer_states[-2], layer_states[-1])
        return layer_states

#优化器的选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
