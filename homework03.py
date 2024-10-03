import torch
import torch.nn as nn
import numpy as np
import random
import json


class TorchModel(nn.Module)
    def  __init__(self, input_size, hidden_size, output_size):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first = True)
        self.classify = nn.Linear(vector_dim, sentence_length + 1)
        self.loss = nn.functional.cross_entropy


    def forward(self, x):
        x= self.embedding(x)
