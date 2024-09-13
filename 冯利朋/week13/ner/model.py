import torch.nn as nn
from transformers import BertModel
from torch.optim import Adam, SGD
from transformers import AutoModelForTokenClassification
from config import Config
TorchModel = AutoModelForTokenClassification.from_pretrained(Config["pretrain_model_path"], num_labels=Config['class_num'])


#
# class TorchModel(nn.Module):
#     def __init__(self, config):
#         super(TorchModel, self).__init__()
#         hidden_size = config['hidden_size']
#         vocab_size = config['vocab_size'] + 1
#         class_num = config['class_num']
#         self.use_bert = config['use_bert']
#         self.embedding = nn.Embedding(vocab_size, hidden_size)
#         if self.use_bert:
#             self.encoder = BertModel.from_pretrained(config['pretrain_model_path'], return_dict=False)
#             hidden_size = self.encoder.config.hidden_size
#         else:
#             self.encoder = nn.LSTM(hidden_size, hidden_size, bidirectional=True,batch_first=True)
#             hidden_size = hidden_size * 2
#         self.classify = nn.Linear(hidden_size, class_num)
#         self.dropout = nn.Dropout(0.2)
#         self.loss = nn.CrossEntropyLoss(ignore_index=-1)
#
#     def forward(self, x, target=None):
#         if self.use_bert:
#             x = self.encoder(x)
#         else:
#             x = self.embedding(x)
#             x = self.encoder(x)
#         if isinstance(x, tuple):
#             x = x[0]
#         x = self.dropout(x)
#         y_pred = self.classify(x)
#         if target is not None:
#             return self.loss(y_pred.view(-1, y_pred.shape[-1]), target.view(-1))
#         else:
#             return y_pred

def choose_optimizer(config, model:TorchModel):
    optimizer = config['optimizer']
    learning_rate = config['learning_rate']
    if optimizer == 'adam':
        return Adam(model.parameters(), lr=learning_rate)
    else:
        return SGD(model.parameters(), lr=learning_rate)





