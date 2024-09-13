import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel

"""
建立网络模型结构
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        output_size = config["output_size"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = config["use_bert"]
        self.emb = nn.Embedding(vocab_size + 1, hidden_size, padding_idx=0)
        if model_type == 'rnn':
            self.encoder = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                                  batch_first=True)
        elif model_type == 'lstm':
            # 双向lstm，输出的是 hidden_size * 2(num_layers 要写2)
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers)
        elif self.use_bert:
            self.encoder = BertModel.from_pretrained(config["bert_model_path"])
            # 需要使用预训练模型的hidden_size
            hidden_size = self.encoder.config.hidden_size
        elif model_type == 'cnn':
            self.encoder = CNN(config)
        elif model_type == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif model_type == "bert_lstm":
            self.encoder = BertLSTM(config)
            # 需要使用预训练模型的hidden_size
            hidden_size = self.encoder.config.hidden_size

        self.classify = nn.Linear(hidden_size, output_size)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.functional.cross_entropy  # loss采用交叉熵损失

    def forward(self, x, y=None):
        if self.use_bert:
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
        # 池化层
        if self.pooling_style == "max":
            # shape[1]代表列数，shape是行和列数构成的元组
            self.pooling_style = nn.MaxPool1d(x.shape[1])
        elif self.pooling_style == "avg":
            self.pooling_style = nn.AvgPool1d(x.shape[1])
        x = self.pooling_style(x.transpose(1, 2)).squeeze()

        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred, y.squeeze())
        else:
            return y_pred


# 定义孪生网络  （计算两个句子之间的相似度）
class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = TorchModel(config)
        # 使用的是cos计算
        # self.loss = nn.CosineEmbeddingLoss()
        # 使用triplet_loss
        self.triplet_loss = self.cosine_triplet_loss

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    # 3个样本  2个为一类 另一个一类 计算triplet loss
    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        return torch.mean(diff[diff.gt(0)])  # greater than

    # 使用triplet_loss
    def forward(self, sentence1, sentence2=None, sentence3=None, margin=None):
        vector1 = self.sentence_encoder(sentence1)
        # 同时传入3 个样本
        if sentence2 is None:
            if sentence3 is None:
                return vector1
            # 计算余弦距离
            else:
                vector3 = self.sentence_encoder(sentence3)
                return self.cosine_distance(vector1, vector3)
        else:
            vector2 = self.sentence_encoder(sentence2)
            if sentence3 is None:
                return self.cosine_distance(vector1, vector2)
            else:
                vector3 = self.sentence_encoder(sentence3)
                return self.triplet_loss(vector1, vector2, vector3, margin)

    # CosineEmbeddingLoss
    # def forward(self,sentence1, sentence2=None, target=None):
    #     # 同时传入两个句子
    #     if sentence2 is not None:
    #         vector1 = self.sentence_encoder(sentence1)  # vec:(batch_size, hidden_size)
    #         vector2 = self.sentence_encoder(sentence2)
    #         # 如果有标签，则计算loss
    #         if target is not None:
    #             return self.loss(vector1, vector2, target.squeeze())
    #         # 如果无标签，计算余弦距离
    #         else:
    #             return self.cosine_distance(vector1, vector2)
    #     # 单独传入一个句子时，认为正在使用向量化能力
    #     else:
    #         return self.sentence_encoder(sentence1)


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


if __name__ == "__main__":
    from config import Config

    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    l = torch.LongTensor([[1], [0]])
    y = model(s1, s2, l)
    print(y)
