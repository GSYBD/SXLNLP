import torch.nn as nn
from transformers import BertModel
from torch.optim import Adam, SGD
from transformers import AutoModelForTokenClassification
from config import Config
TorchModel = AutoModelForTokenClassification.from_pretrained(Config["pretrain_model_path"], num_labels=Config['class_num'])




def choose_optimizer(config, model:TorchModel):
    optimizer = config['optimizer']
    learning_rate = config['learning_rate']
    if optimizer == 'adam':
        return Adam(model.parameters(), lr=learning_rate)
    else:
        return SGD(model.parameters(), lr=learning_rate)





