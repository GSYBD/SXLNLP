import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF
from torch.optim import Adam, SGD
from config import Config

"""

    建立网络模型结构

"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.use_crf = config['use_crf']
        self.encoder = BertModel.from_pretrained(config['pretrain_model_path'], return_dict=False)
        self.hidden_size = self.encoder.config.hidden_size

        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.classify = nn.Linear(self.hidden_size, config['class_num'])
        self.crf_layer = CRF(config['class_num'], batch_first=True)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, target=None):
        x, _ = self.encoder(x)
        # (batch_size * max_length) -> [(batch_size * max_length * hidden_size), (batch_size * hidden_size)]
        # x, _ = self.lstm(x)
        y = self.classify(x)
        # (batch_size * max_length * hidden_size) -> (batch_size, max_length, class_num)
        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(y, target, mask, reduction='mean')
            else:
                return self.loss(y.view(-1, y.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(y)
            else:
                return y


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    torch_model = TorchModel(Config)
    x_ = torch.LongTensor([[265, 3778, 27, 185, 868, 1803, 1320, 1163, 2795, 525, 597, 232,
                           489, 2609, 2769, 2025, 454, 969, 3004, 3881, 2769, 1192, 552, 2344,
                           1508, 1418, 3574, 727, 165, 1117, 145, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0]])
    y_ = torch.LongTensor([[8, 8, 8, 1, 5, 5, 5, 8, 3, 7, 0, 4, 8, 8, 8, 8, 8, 8,
                          8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
    """
    {
      "B-LOCATION": 0,
      "B-ORGANIZATION": 1,
      "B-PERSON": 2,
      "B-TIME": 3,
      "I-LOCATION": 4,
      "I-ORGANIZATION": 5,
      "I-PERSON": 6,
      "I-TIME": 7,
      "O": 8
    }
    """
    print(torch.argmax(torch_model(x_), dim=2))
    print(torch_model(x_, y_))
    Config['use_crf'] = True
    torch_model = TorchModel(Config)
    print(torch_model(x_))
    print(torch_model(x_, y_))
