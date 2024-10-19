import torch.nn
import torch.nn as nn
from config import Config
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel
from torch.optim import Adam, SGD
from torchcrf import CRF


# TorchModel = AutoModelForSequenceClassification.from_pretrained(Config["pretrain_model_path"])

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.pooling_layer = None
        class_num = config["class_num"]
        self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        hidden_size = self.encoder.config.hidden_size
        self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, target=None):
        x, _ = self.encoder(x)
        predict = self.classify(x)
        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
