"""
构建一个 用RNN实现的 判断某个字符的位置 的任务

5 分类任务 判断 a出现的位置 返回index +1 or -1
"""
import random

import numpy as np
import torch
from torch import nn
import torch.utils.data as Data


class TorchModel(nn.Module):
    def __init__(self, sentence_length, vocab, hidden_size):
        super(TorchModel, self).__init__()
        self.emb = nn.Embedding(len(vocab) + 1, hidden_size, padding_idx=0)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.pool = nn.MaxPool1d(sentence_length)
        self.fc = nn.Linear(hidden_size, 5)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.emb(x)
        x, h = self.rnn(x)
        x = self.pool(x.transpose(1, 2)).squeeze()
        y_pred = self.fc(x)
        if y is not None:
            return self.loss(y_pred, y.view(-1))
        else:
            return y_pred


def build_simple(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    str = ''.join(x)
    if 'x' in str and 'y' in str and 'z' in str:
        y = 0
    elif 'X' in str and 'Y' in str and 'Z' in str:
        y = 1
    elif 'a' in str and 'b' in str and 'c' in str:
        y = 2
    elif 'A' in str and 'B' in str and 'C' in str:
        y = 3
    else:
        y = 4
    x = [vocab.get(x[i]) for i in range(sentence_length)]
    return x, y


def build_dataset(train_simple, vocab, sentence_length):
    X = []
    Y = []
    for i in range(train_simple):
        x, y = build_simple(vocab, sentence_length)
        X.append(x)
        Y.append([y])

    return torch.LongTensor(X), torch.LongTensor(Y)


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    # vocab['unk'] = len(vocab) + 1
    return vocab


def evaluate(model, vocab, sentence_length):
    x, y = build_dataset(100, vocab, sentence_length)
    model.eval()
    correct, error = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_t, y_p in zip(y, y_pred):
            if int(y_t) == int(torch.argmax(y_p)):
                correct += 1
            else:
                error += 1
    print("正确预测个数：%d / %d, 正确率：%f" % (correct, correct + error, correct / (correct + error)))
    return correct / (correct + error)



def main():
    batch_size = 20
    lr = 0.002
    train_simple = 5000
    hidden_size = 64
    vocab = build_vocab()
    epoch_size = 10
    sentence_length = 20
    # build model
    model = TorchModel(sentence_length, vocab, hidden_size)
    # 優化器
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # 訓練的數據
    X, Y = build_dataset(train_simple, vocab, sentence_length)
    # 分割數據
    dataset = Data.TensorDataset(X, Y)
    data_item = Data.DataLoader(dataset, batch_size, shuffle=True)
    for epoch in range(epoch_size):
        epoch_loss = []
        model.train()
        for x, y_true in data_item:
            loss = model(x, y_true)
            loss.backward()
            optim.step()
            optim.zero_grad()
            epoch_loss.append(loss.item())
        print("第%d轮 loss = %f" % (epoch + 1, np.mean(epoch_loss)))
        # evaluate
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
    # save model
    torch.save(model.state_dict(), "model_work3.pt")
    return

def predict(str):
    vocab = build_vocab()
    x = [vocab.get(char) for char in str]

    sentence_length = 10
    hidden_size = 64
    model = TorchModel(sentence_length,vocab,hidden_size)
    # 读取路径
    model.load_state_dict(torch.load("model_work3.pt"))
    # 测试模式
    model.eval()
    with torch.no_grad():  # 不计算梯度
        # 模型预测的结果
        result = model.forward(torch.LongTensor([x]))

        print("预测类别：%s ",torch.argmax(result))


if __name__ == '__main__':
    # main()
    predict("XabxyYzaba")

