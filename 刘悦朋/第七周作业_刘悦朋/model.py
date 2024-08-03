import torch.nn as nn
from config import Config
import loader
import torch
from torch.optim import Adam, SGD
from transformers import BertModel

"""

    建立网络模型结构

"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config['hidden_size']
        vocab_size = config['vocab_size'] + 1
        model_type = config['model_type']
        num_layers = config['num_layers']
        class_num = config['class_num']
        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if model_type == 'lstm':
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == 'bert':
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config['pretrain_model_path'], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        elif model_type == 'bert_lstm':
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size

        self.classify = nn.Linear(hidden_size, 1)
        self.activation = nn.functional.sigmoid
        self.pooling_style = config['pooling_style']
        self.loss = nn.functional.mse_loss

    # 当输入真实标签, 返回loss值; 无真实标签, 返回预测值
    def forward(self, x, target=None):
        if self.use_bert:
            # bert返回的结果是 (sequence_output, pooler_output)
            # sequence_output: batch_size, max_len, hidden_size
            # pooler_output: batch_size, hidden_size
            x = self.encoder(x)
        else:
            x = self.embedding(x)
            x = self.encoder(x)

        # if x is tuple
        # RNN类的模型会同时返回隐单元向量, 我们只取序列结果
        if isinstance(x, tuple):
            x = x[0]
        # 可以采用pooling的方式得到句向量
        if self.pooling_style == 'max':
            self.pooling_layer = nn.MaxPool1d(kernel_size=x.shape[1])  # sen_len
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1, 2)).squeeze()  # input shape:(batch_size, sen_len, input_dim)

        # 也可以直接使用序列最后一个位置的向量
        # x = x[:, -1, :]
        predict = self.classify(x)  # input shape:(batch_size, input_dim)
        predict = self.activation(predict)

        if target is not None:
            return self.loss(predict, target)
        else:
            return predict


class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(config['pretrain_model_path'], return_dict=False)
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]
        x, _ = self.rnn(x)
        return x


# 优化器的选择
def choose_optimizer(config, model):
    optimizer = config['optimizer']
    learning_rate = config['learning_rate']
    if optimizer == 'adam':
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == 'SGD':
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    train_data = loader.load_data(Config['train_data_path'], Config)
    model = TorchModel(Config)
    x = torch.LongTensor([[2173, 4413, 3720, 1861, 4276, 1419, 220, 1011, 974, 4612, 637, 1419,
                           637, 1011, 4612, 1011, 3747, 4606, 4606, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0]])
    print(model(x))
