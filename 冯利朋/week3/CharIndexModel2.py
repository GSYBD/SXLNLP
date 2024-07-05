"""
使用RNN判断某一个字出现的位置,此例子是判断"中"字的位置
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim



# 1.创建词汇表
def build_vocab():
    chars = "rsqpo在家中i红kejyw国nuvxgdlcahfmtbz"
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab) + 1
    return vocab


# 2.创建训练数据集
def build_dataset(vocab, sample_num, sentence_length):
    X = []
    Y = []
    for _ in range(sample_num):
        x = list(vocab.keys())
        x.remove('unk')
        random.shuffle(x)
        x = x[:sentence_length]
        y = x.index('中')
        x = [vocab.get(char, vocab['unk']) for char in x]
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)


# 3.创建模型
class TorchModel(nn.Module):
    def __init__(self, vocab, char_dim, sentence_length, hidden_size,num_class):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, char_dim, padding_idx=0)
        self.nb = nn.LayerNorm(char_dim)
        self.dropout = nn.Dropout(0.3)
        self.rnn_layer = nn.LSTM(input_size=char_dim, hidden_size=hidden_size, batch_first=True)
        self.classify = nn.Linear(hidden_size,num_class)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x) # batch_size,sen_len ----- batch_size,sen_len,char_dim
        x = self.nb(x)
        x = self.dropout(x)
        x, (ht,ct) = self.rnn_layer(x) # batch_size,sen_len,char_dim ----- batch_size,sen_len,hidden_size
        y_pred = self.classify(ht.squeeze()) # batch_size,sen_len,hidden_size ---- batch_size,sen_len,num_class
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def main():
    epoch_num = 10
    batch_size = 20
    char_dim = 50
    hidden_size = 100
    vocab = build_vocab()
    sentence_length = len(vocab) - 1
    sample_num = 5000
    num_class = sentence_length
    model = TorchModel(vocab, char_dim, sentence_length, hidden_size,num_class)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    dataset_x, dataset_y = build_dataset(vocab,sample_num,sentence_length)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(sample_num // batch_size):
            x = dataset_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = dataset_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        print("第%d轮,loss=%f" % (epoch + 1,np.mean(watch_loss)))
    torch.save(model.state_dict(), 'model.pth')

def predict(model_path,input_str):
    vocab = build_vocab()
    char_dim = 50
    hidden_size = 100
    sentence_length = len(vocab) - 1
    num_class = sentence_length
    model = TorchModel(vocab, char_dim, sentence_length, hidden_size,num_class)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    X = []
    for sentence in input_str:
        x = [vocab.get(char) for char in sentence]
        X.append(x)
    X = torch.LongTensor(X)
    with torch.no_grad():
        y_pred = model(X)
    for y_p, y_t in zip(y_pred, input_str):
        i = y_t.index('中')
        print("正确位置:%d,预测位置:%d,是否正确:%s" % (i, torch.argmax(y_p), (torch.argmax(y_p) == i)))


if __name__ == '__main__':
    # main()
    input_str = ['fegdkm红uraov在qjz国txi家中wyhcbsnlp',
                 'rdt红ngye国cijzxlhwof在k中ubmaqsv家p',
                 'syi中xwcudpaqgmok国ftzjre红blnv在家h',
                 '家国f中jgcvitdulqn在红rboxzehawpyksm',
                 'nmbkuzsl国rcq中tjyif在xaoghpewd家v红',
                 'hald红cynwe国gjbzfxpksqr家中oiu在tmv',
                 'n国s在yhpklxe红家qzua中irbwgojtvdmcf',
                 '中红yktdov国sngmhuarlz在feqjib家pcwx',
                 'wtmnsgyoxluq红faich中dpjr在k国家bezv',
                 '在qvn红iejkphdobmw国tx中gcrualyzfs家']

    predict('model.pth', input_str)


