# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""
实现一个基于交叉熵的多分类任务，
任务：x是一个4维向量，如果第1个数+第2个数 > 第3个数+第4个数，则为正样本，反之为负样本
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 2)
        # 激活函数 sigmoid归一化函数
        # self.activation = torch.sigmoid
        # 损失函数 交叉熵
        self.loss = nn.functional.cross_entropy

    # 这个函数有两个功能：1.计算loss值，2.计算预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)
        # y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


# 随机生成一个4维向量,如果第1个数+第2个数 > 第3个数+第4个数，则为正样本，反之为负样本
def build_sample():
    # 随机生成一个4维向量
    x = np.random.random(4)
    # 如果第1个数+第2个数 > 第3个数+第4个数，则为正样本，反之为负样本
    if x[0] + x[1] > x[2] + x[3]:
        y = 1
    else:
        y = 0
    return x, y


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        a, b = build_sample()
        X.append(a)
        Y.append(b)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
def evaluate(model):
    model.eval()
    test_sample_num = 10
    correct, wrong = 0, 0
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    with torch.no_grad():
        # 模型预测
        y_pred = model(x)
        # print("y_pred:", y_pred)
        # 预测值与真实标签进行对比
        for y_p, y_t in zip(y_pred, y):
            # print(y_p, y_t)
            p = y_p[0] <= y_p[1]
            if p == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    # 训练轮数
    epoch_num = 20
    # 每次训练样本个数
    batch_size = 500
    # 训练样本总数
    sample_num = 10000
    # 输入维度
    input_size = 4
    # 学习率
    lr = 0.01
    # 网络模型
    model = TorchModel(input_size)
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # 结果集
    log = []
    # 训练集
    train_x, train_y = build_dataset(sample_num)
    print("train_x:", train_x)
    print("train_y:", train_y)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(sample_num // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            # 计算loss值
            loss = model(x, y)
            # 计算梯度
            loss.backward()
            #  更新权重
            optim.step()
            # 梯度清零
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # 测试本轮模型结果
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model2.pth")
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()


def predict(model_path, input_vec):
    model = TorchModel(4)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))

    for x, y in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (x, y[0] <= y[1], y[1]))


if __name__ == "__main__":
    #main()
    test_vec = [
        [0.5, 0.2, 0.33, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 0.2, 0.1, 0.4],
        [0.9, 0.6, 0.3, 0.8],
        [0.9, 0.2, 0.3, 0.8],
        [0.9, 0.6, 1.3, 0.8],
        [1.9, 0.2, 0.31, 0.8],
        [0.9, 0.6, 2.3, 0.8],
        [0.9, 3.2, 0.3, 0.8],
    ]
    predict("model2.pth", test_vec)
