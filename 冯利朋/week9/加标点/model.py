import torch.nn as nn
from transformers import BertModel
import torch
class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config['hidden_size']
        vocab_size = config['vocab_size'] + 1
        class_num = config['class_num']
        self.use_bert = config['use_bert']
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if self.use_bert:
            self.encoder = BertModel.from_pretrained(config['pretrain_model_path'], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        else:
            self.encoder = nn.LSTM(hidden_size, hidden_size, bidirectional=True, batch_first=True, num_layers=2)
            hidden_size = 2 * hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classify = nn.Linear(hidden_size, class_num)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
    def forward(self, x, attention_mask, target=None):
        if self.use_bert:
            x = self.encoder(x, attention_mask=attention_mask)
        else:
            x = self.embedding(x)
            x = self.encoder(x)
        if isinstance(x, tuple):
            x = x[0]
        x = self.dropout(x)
        pred = self.classify(x)
        if target is not None:
            return self.loss(pred.view(-1,pred.shape[-1]), target.view(-1))
        else:
            return pred

def choose_optimizer(config, model):
    learning_rate = config['learning_rate']
    optimizer = config['optimizer']
    if optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        return torch.optim.SGD(model.parameters(), lr=learning_rate)

