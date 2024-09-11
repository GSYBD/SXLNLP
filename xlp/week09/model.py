import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel
from torchcrf import CRF

"""
建立网络模型结构
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        output_size = config["output_size"]
        self.model_type = config["model_type"]
        num_layers = config["num_layers"]
        # self.use_bert = config["use_bert"]
        self.use_crf = config["use_crf"]

        self.emb = nn.Embedding(vocab_size + 1, hidden_size, padding_idx=0)
        if self.model_type == 'rnn':
            self.encoder = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                                  batch_first=True)
        elif self.model_type == 'lstm':
            # 双向lstm，输出的是 hidden_size * 2(num_layers 要写2)
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
            hidden_size = hidden_size * 2

        elif self.model_type == 'bert':
            self.encoder = BertModel.from_pretrained(config["bert_model_path"])
            # 需要使用预训练模型的hidden_size
            hidden_size = self.encoder.config.hidden_size
        elif self.model_type == 'cnn':
            self.encoder = CNN(config)
        elif self.model_type == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif self.model_type == "bert_lstm":
            self.encoder = BertLSTM(config)
            # 需要使用预训练模型的hidden_size
            hidden_size = self.encoder.config.hidden_size
        self.classify = nn.Linear(hidden_size, output_size)
        self.pooling_style = config["pooling_style"]
        self.crf_layer = CRF(output_size, batch_first=True)
        # 添加（） 使用torch.nn
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  # loss采用交叉熵损失

    def forward(self, x, y=None):
        if self.model_type == 'bert':
            # 输入x为[batch_size, seq_len]
            # bert返回的结果是 (sequence_output, pooler_output)
            # sequence_output:batch_size, max_len, hidden_size
            # pooler_output:batch_size, hidden_size
            x = self.encoder(x)[0]
        else:
            x = self.emb(x)
            x = self.encoder(x)
        # 判断x是否是tuple
        if isinstance(x, tuple):
            x = x[0]

        # # 池化层
        # if not self.use_crf:
        #     if self.pooling_style == "max":
        #         # shape[1]代表列数，shape是行和列数构成的元组
        #         self.pooling_style = nn.MaxPool1d(x.shape[1])
        #     elif self.pooling_style == "avg":
        #         self.pooling_style = nn.AvgPool1d(x.shape[1])
        #     x = self.pooling_style(x.transpose(1, 2)).squeeze()

        y_pred = self.classify(x)
        if y is not None:
            # 是否使用crf：
            if self.use_crf:
                mask = y.gt(-1)
                return - self.crf_layer(y_pred, y, mask, reduction="mean")
            else:
                # (number, class_num), (number)
                # 返回的y_pred是(batch_size, max_len, class_num)
                # y是(batch_size, max_len)
                # 所以计算loss 应该 展开为y_pred[batch_size*max_len,class_num] y[batch_size*max_len]
                return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(y_pred)
            else:
                return y_pred

# 优化器的选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["lr"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


# 定义CNN模型
class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1) / 2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x):  # x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)


# 定义GatedCNN模型
class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)

    # 定义前向传播函数 比普通cnn多了一次sigmoid 然后互相卷积
    def forward(self, x):
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b)


# 定义BERT-LSTM模型
class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_model_path"], return_dict=False)
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]
        x, _ = self.rnn(x)
        return x

# if __name__ == "__main__":
#     from config import Config
#
#     Config["output_size"] = 2
#     Config["vocab_size"] = 20
#     Config["max_length"] = 5
#     Config["model_type"] = "bert"
#     Config["use_bert"] = True
#     # model = BertModel.from_pretrained(Config["bert_model_path"], return_dict=False)
#     x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
#     # sequence_output, pooler_output = model(x)
#     # print(x[1], type(x[2]), len(x[2]))
#
#     model = TorchModel(Config)
#     label = torch.LongTensor([0,1])
#     print(model(x, label))
