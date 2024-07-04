#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

"""

实现一个基于交叉熵的多分类任务，判断最大的那个数
规律：x是一个5维向量，需要将x的每一个数进行比较，如果第一个数最大就是第一类，如果第二个数最大就是第二类，以此类推对5个向量进行比较分类。

"""


# 定义模型
class MaxIndexClassifier(nn.Module):
    def __init__(self, input_size):
        super(MaxIndexClassifier, self).__init__()
        self.fc = nn.Linear(input_size, 5)  # 线性层 5层
        self.activation = nn.Softmax()  # Softmax归一化函数
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        out = self.fc(x)
        y_pred = self.activation(out)
        if y is not None:
            return self.loss(y_pred, y)  # 计算模型预测值和真实值之间的损失值，并将其作为函数的返回值
        else:
            return y_pred  # 输出预测结果


# 生成单个样本
def generate_sample():
    X = np.random.rand(5)  # 随机生成一个5维向量
    y = np.argmax(X)  # 找出最大值的索引作为标签
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# 生成数据集
def generate_data(num_samples):
    X = []
    y = []
    for i in range(num_samples):
        sample, label = generate_sample()
        X.append(sample)
        y.append(label)
    return torch.stack(X), torch.tensor(y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = generate_data(test_sample_num)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, error = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        _, predicted = torch.max(y_pred, 1)
        for y_p, y_t in zip(predicted, y):  # 与真实标签进行对比
            if y_p == y_t:
                correct += 1  # 预测正确
            else:
                error += 1
    accuracy = correct / (correct + error)
    print(f"正确预测个数：{correct}, 正确率：{accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    input_size = 5  # 输入向量维度
    num_samples = 5000  # 样本数
    num_epochs = 100  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    learning_rate = 0.001  # 学习率
    # 创建模型
    model = MaxIndexClassifier(input_size)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 生成训练数据
    x_train, y_train = generate_data(num_samples)
    log = []

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        watch_loss = []

        for batch_index in range(num_samples // batch_size):
            x = x_train[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = y_train[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss: {np.mean(watch_loss):.4f}")
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model.pt")

    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


def predict(model_path, input_vecs):
    input_size = 5
    model = MaxIndexClassifier(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        for input_vec in input_vecs:
            result = model.forward(torch.FloatTensor([input_vec]))
            vec = np.array(input_vec)
            _, predicted_class = torch.max(result, 1)
            print(f"输入：{vec}, 预测类别：{predicted_class.item()}")  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.07889086, 1.15229675, 0.31082123, 0.03504317, 4.18920843],
                [0.94963533, 2.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.78797868, 0.67482528, 3.13625847, 0.34675372, 0.19871392],
                [0.79349776, 0.59416669, 0.92579291, 4.41567412, 4.1358894]]
    predict("model.pt", test_vec)
