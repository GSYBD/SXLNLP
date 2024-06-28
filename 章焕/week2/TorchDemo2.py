#程序中链接了多个 OpenMP 运行时库的副本
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import math

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，找出其中最大的一个数 ，输入一个五维向量，输出最大数的位数，如：输入[3 4 5 6 7] 输出[4]

"""


class TorchModel2(nn.Module):
    def __init__(self, input_size):
        super(TorchModel2, self).__init__()
        self.layer = nn.Linear(input_size, input_size)  # 线性层
        # self.activation = nn.Softmax(dim=0)  # Softmax
        self.loss = nn.functional.cross_entropy  # loss函数采用 交叉熵 计算

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        #x = self.layer(x)  # (batch_size, input_size) -> (batch_size, 1)
        #x = self.activation(x)
        y_pred = self.layer(x)  # (batch_size, 1) -> (batch_size, 1)
        #print(y_pred)
        #print(y.squeeze().long())
        if y is not None:
            return self.loss(y_pred, y.squeeze().long())  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，输出它的最大的数的位数
def build_sample():
    x = np.random.random(5)
    max_index = max(enumerate(x), key=lambda x: x[1])[0]
    return x, max_index


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)


def main():
    # 配置参数  训练20轮 总样本5000 每个样本250个 共20个样本进行训练
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel2(input_size)
    # 损失函数使用 交叉熵
    # ce_loss = nn.CrossEntropyLoss()
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()  #将模型设置为训练模型
        watch_loss = []
        for batch_index in range(train_sample // batch_size):  # // 除法取整
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
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


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    y = y.squeeze()
    print("A类样本数量：%d, B类样本数量：%d, C类样本数量：%d, D类样本数量：%d, E类样本数量：%d" % (
    y.tolist().count(0), y.tolist().count(1), y.tolist().count(2), y.tolist().count(3), y.tolist().count(4)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d / %d, 正确率：%f" % (correct, test_sample_num, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel2(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式 将模型设置为计算模型
    with torch.no_grad():  # 不计算梯度
        result = model(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, torch.argmax(res), res))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.pt", test_vec)

    # x = np.random.random(5)
    # max_index = max(enumerate(x), key=lambda x: x[1])[0]
    # print(x)
    # print(max_index+1)
