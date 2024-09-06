
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained('./bert-base-chinese', return_dict=False)
        self.bertokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
        self.classify = nn.Linear(768, len(self.bertokenizer))
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y, attention_mask):
        x = x.squeeze(1)
        y = y.squeeze(1)
        if y is not None:
            x, _ = self.bert(x, encoder_attention_mask=attention_mask)
            y_pred = self.classify(x)
            y_pred = y_pred.view(-1, y_pred.size(-1))
            loss = self.loss(y_pred, y.view(-1), ignore_index=-100)
            return loss
        else:
            return x