"""
5分类任务，如果a在位置0，类别为0
         如果a在位置1，类别为1
         如果a在位置2，类别为2
         如果a在位置3，类别为3
         如果文本不包含a，类别为4，即其他
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import random
import json
import matplotlib.pyplot as plt

sentence_length = 4  # 类似abcd的长度为4
vector_dim = 5  # enbeding字符长度
hidden_size = 5
output_size = 5
epoch_number = 20
batch_size = 20
learning_rate = 0.005
train_sample = 5000
eval_sample = 100


class TorchModel(nn.Module):
    def __init__(self, vocabulary, sentence_length, vector_dim, output_size):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocabulary), vector_dim)
        # self.pool = nn.AvgPool1d(sentence_length)
        self.rnn = nn.RNN(vector_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)
        # x = x.transpose(1, 2)
        # x = self.pool(x)
        # h = x.squeeze()
        output,h = self.rnn(x)
        h = h.squeeze()
        y_pre = self.linear(h)
        if y is not None:
            return self.loss(y_pre, y)
        else:
            return y_pre


# 创建字典
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab["unk"] = len(vocab)
    return vocab


vocabulary = build_vocab()


# 创建样本
def sample(vocabulary, sentence_length):
    x = random.choices((list(vocabulary.keys())), k=sentence_length)
    if x[0] == "a":
        y = 0
    elif x[1] == "a":
        y = 1
    elif x[2] == "a":
        y = 2
    elif x[3] == "a":
        y = 3
    else:
        y = 4

    x = [vocabulary.get(word, vocabulary["unk"]) for word in x]
    return x, y


# 创建样本集
def sample_set(vocabulary, sentence_length, train_sample):
    X = []
    Y = []
    for i in range(train_sample):
        x, y = sample(vocabulary, sentence_length)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(np.array(X)), torch.LongTensor(np.array(Y))


def evaluate(model, vocabulary):
    correct = 0
    X_eval, Y_eval = sample_set(vocabulary, sentence_length, eval_sample)

    model.eval()
    with torch.no_grad():
        Y_pre = model(X_eval)
        Y_eval = Y_eval.numpy().tolist()
        Y_pre = Y_pre.numpy().tolist()
        for y_p, y_t in zip(Y_pre, Y_eval):
            if y_p.index(max(y_p)) == y_t:
                correct += 1

    return correct / eval_sample


def main():
    model = TorchModel(vocabulary, sentence_length, vector_dim, output_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    X, Y = sample_set(vocabulary, sentence_length, train_sample)
    total_loss = []
    eval_set = []
    for i in range(epoch_number):
        watch_loss = []
        model.train()
        for j in range(train_sample // batch_size):
            X_batch = X[j * batch_size:(j + 1) * batch_size]
            Y_batch = Y[j * batch_size:(j + 1) * batch_size]

            loss = model(X_batch, Y_batch)  # 计算损失
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())

        total_loss.append(np.mean(watch_loss))
        eval = evaluate(model, vocabulary)
        eval_set.append(eval)
        print("\n第%d轮的损失为：%f,准确率为：%f" % (i + 1, np.mean(watch_loss), eval))

    torch.save(model.state_dict(), "model.pth")
    plt.plot(total_loss, label="loss")
    plt.plot(eval_set, label="accuary")
    plt.legend()
    plt.show()


def predict(model_path, x_char, vocabulary):
    x_vers = []
    for i in range(len(x_char)):
        x_ver = [vocabulary.get(word, vocabulary["unk"]) for word in x_char[i]]
        x_vers.append(x_ver)

    model = TorchModel(vocabulary, sentence_length, vector_dim, output_size)
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        result = model(torch.LongTensor(x_vers))
    result = result.numpy().tolist()
    for i, result_i in enumerate(result):
        print("输入：%s，预测类别为:%d" % (x_char[i], result_i.index(max(result_i))))


if __name__ == "__main__":
    # main()
    test_strings = ["bbee", "sdfg", "raeg", "abkw"]
    predict("model.pth", test_strings, vocabulary)
