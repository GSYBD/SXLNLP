import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, vocab_size, dim, hidden_size, label_size) -> None:
        super(Model, self).__init__() # remember this
        self.embedding = nn.Embedding(vocab_size, dim)
        self.rnn = nn.RNN(dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, label_size)

    def forward(self, x):
        x = self.embedding(x)
        _, x = self.rnn(x)
        x = x[-1,:,:] # 查看一下
        x = self.linear(x)
        return x


