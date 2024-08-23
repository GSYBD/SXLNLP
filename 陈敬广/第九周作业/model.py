import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel

'''
建立模型结构
'''


class NerModel(nn.Module):
    def __init__(self, config):
        super(NerModel, self).__init__()
        self.config = config
        self.vocab_size = config['vocab_size'] + 1
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.class_num = config['class_num']
        # self.embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0)
        # self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, num_layers=self.num_layers)
        self.bert = BertModel.from_pretrained(self.config['bert_path'],return_dict=False)
        self.fc = nn.Linear(self.hidden_size, self.class_num)
        self.crf = CRF(self.class_num, batch_first=True)
        self.use_crf = config['use_crf']
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, y=None):
        x,_ = self.bert(x)
        y_pred = self.fc(x)
        if y is not None:
            if self.use_crf:
                mask = y.gt(-1)  # 只计算大于-1的标签
                return self.crf(y_pred, y, mask, reduction='mean')
            else:
                return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            if self.use_crf:
                return self.crf.viterbi_decode(y_pred)
            else:
                return y_pred


def choose_optim(config, model):
    optim = config['optimizer']
    learning_rate = config['learning_rate']
    if optim == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        return torch.optim.Adam(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    from config import Config

    model = NerModel(Config)
