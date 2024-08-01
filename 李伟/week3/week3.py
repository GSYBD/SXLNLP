import random

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的机器学习任务
输入一个字符串，根据字符a所在的位置进行分类
"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length,vocab):
        super(TorchModel,self).__init__()
        self.embedding = nn.Embedding(len(vocab),vector_dim)
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        self.classify=nn.Linear(vector_dim,sentence_length+1)
        self.loss=nn.functional.cross_entropy

    def forward(self,x,y=None):
        x=self.embedding(x)
        rnn_out,hidden=self.rnn(x)
        x=rnn_out[:,-1,:]

        y_pred=self.classify(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred

     def build_data(vocab,sentence_length):
         x=random.sample(list(vocab.keys()),sentence_length)
         if "a" in x:
             y=x.index("a")
         else:
             y=sentence_length
         x=[vocab.get(word,vocab['unk']) for word in x]
         return x,y

    def build_dataset(sample_length, vocab, sentence_length):
        dataset_x = []
        dataset_y = []
        for i in range(sample_length):
            x, y = build_sample(vocab, sentence_length)
            dataset_x.append(x)
            dataset_y.append(y)
        return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

    def build_model(vocab, char_dim, sentence_length):
        model = TorchModel(char_dim, sentence_length, vocab)
        return model

    def evaluate(model, vocab, sample_length):
        model.eval()
        x, y = build_dataset(200, vocab, sample_length)
        correct, wrong = 0, 0
        with torch.no_grad():
            y_pred = model(x)
            for y_p, y_t in zip(y_pred, y):
                if int(torch.argmax(y_p)) == int(y_t):
                    correct += 1
                else:
                    wrong += 1
        print("正确预测个数:%d，正确率:%f" % (correct, correct / (correct + wrong)))
        return correct / (correct + wrong)

    def main():
        epoch_nbum=20
        batch_size=40
        train_sample=1000
        char_dim=30
        sentence_length=10
        learning_rate=0.001
        vocab=build_vocab()
        model=build_model(vocab,char_dim,sentence_length)
        optim=torch.optim.Adam(model.parameters(),lr=learning_rate)
        log=[]
