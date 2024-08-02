import torch
import torch.nn as nn
from transformers import BertModel

class TorchModel(nn.Module):
    def __init__(self, input_dim):
        super(TorchModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese", return_dict=False)
        self.classify = nn.Linear(input_dim, 2)
        self.activation = torch.sigmoid
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        sequence_output, pooler_output = self.bert(x)
        x = self.classify(pooler_output)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(torch.FloatTensor(y_pred), torch.LongTensor(y.squeeze()))
        else:
            return torch.FloatTensor(y_pred)
