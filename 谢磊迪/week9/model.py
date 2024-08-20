import torch
from torch import nn
from create_datas import load_datas, config
from torchcrf import CRF
import torch.optim as optim
import black
from  transformers import BertTokenizer, BertModel
import warnings

warnings.filterwarnings("ignore")
"""
模型是
"""


class Ner_Model(torch.nn.Module):
    def __init__(self, config):
        super(Ner_Model, self).__init__()
        # vocab_size = config["vocab_size"]
        self.optim = config["optim"]
        self.lr = config["lr"]
        num_layers = config["num_layers"]
        hidden_size = config["embedding_dim"]
        schema_size = config["schema_size"]
        # self.embedding = nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=hidden_size, padding_idx=0)
        # self.lstm = nn.LSTM(
        #     input_size=hidden_size,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     bidirectional=True,
        #     batch_first=True,
        # )
        self.bert =BertModel.from_pretrained(config['bert_path'],return_dict = False)
        self.linear = nn.Linear(self.bert.config.hidden_size, schema_size)
        self.crf_layer = CRF(schema_size, batch_first=True)
        self.user_crf = config["use_crf"]
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, true_y=None):
        # x = self.embedding(x)
        # x, _ = self.lstm(x)
        x,_ = self.bert(x)
        predict_y = self.linear(x)
        # print('predict_y  ok')
        if true_y is not None:
            if self.user_crf:
                mask = true_y.gt(-1)
                return -self.crf_layer(predict_y, true_y, mask, reduction="mean")
            else:
                return self.loss(predict_y.reshape(-1, predict_y.shape[2]), true_y.reshape(-1))
        else:
            if self.user_crf:
                return self.crf_layer.decode(predict_y)
            else:
                return predict_y


def choice_optim(model, config):
    if config["optim"] == "Adam":
        return torch.optim.Adam(model.parameters(), lr=config["lr"])
    elif config["optim"] == "SGD":
        return torch.optim.SGD(model.parameters(), lr=config["lr"])


if __name__ == "__main__":
    dl = load_datas(config)
    for i in dl:
        x = i[0]
        y = i[1]
        model = Ner_Model(config)
        model.forward(x, true_y=y)
        print(1)
