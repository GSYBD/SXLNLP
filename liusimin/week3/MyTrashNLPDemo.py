#coding:utf8

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from random import shuffle
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

"""


def build_vocab(chars):
    """
    字符集随便挑了一些字，实际上还可以扩充
    为每个字生成一个标号
    {"a":1, "b":2, "c":3...}
    abc -> [1,2,3]
    :return:
    """
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

chars = "opqrstuvwxyz"  #字符集
vocab = build_vocab(chars)


#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample():
    x = shuffle_string(chars)
    y = cal_label(x)
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y


def shuffle_string(str):
    strlist = list(str)
    shuffle(strlist)
    return ''.join(strlist)


def cal_label(x: str):
    return x.find('x')


#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample()
        dataset_x.append(x)
        dataset_y.append(y)
    return TensorDataset(torch.LongTensor(dataset_x), torch.LongTensor(dataset_y))


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, tester=False):
        super(RNNModel, self).__init__()
        self.tester = tester
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    def forward(self, x, y=None):
        # 原始数据 (batch_size, sen_len) 32*6
        self.print(f"x:{x.shape}")
        x = self.embedding(x)
        self.print(f"embedding:{x.shape}")
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # 取序列最后一个输出
        self.print(f"rnn x:{x.shape},_:{_}")
        y_pred = self.fc(x)
        if y is not None:
            self.print(f"rnn y_pred:{y_pred.shape}, y:{y.shape}")
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred

    def print(self, x):
        if self.tester:
            print(x)


#测试代码
#用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    dataset = build_dataset(100)   #建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(dataset.tensors[0])      #模型预测
        for y_p, y_t in zip(y_pred, dataset.tensors[1]):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    hidden_size = 64  # RNN隐藏层大小
    sentence_length = len(chars)  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立模型
    model = RNNModel(len(vocab), char_dim, hidden_size, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    dataset = build_dataset(train_sample)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for inputs, labels in dataloader:
            optim.zero_grad()    #梯度归零
            loss = model(inputs, labels)  # 计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    # torch.save(model.state_dict(), "model.pth")
    # 保存词表
    # writer = open("vocab.json", "w", encoding="utf8")
    # writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    # writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = RNNModel(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i])) #打印结果


def test():
    # 配置参数
    epoch_num = 1  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 32  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    hidden_size = 64  # RNN隐藏层大小
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立模型
    model = RNNModel(len(vocab), char_dim, hidden_size, sentence_length, tester=True)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    dataset = build_dataset(train_sample)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for inputs, labels in dataloader:
            optim.zero_grad()  # 梯度归零
            loss = model(inputs, labels)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])


if __name__ == "__main__":
    # test()
    main()
    # test_strings = ["fnvfee", "wzsdfg", "rqwdeg", "nakwww"]
    # predict("model.pth", "vocab.json", test_strings)
