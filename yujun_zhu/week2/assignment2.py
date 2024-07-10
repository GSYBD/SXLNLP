# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import os

"""

基于pytorch框架编写模型训练

实现一个基于交叉熵的多分类任务，任务可以自拟。


1st biggest, class 5
2nd biggest, class 4
3rd biggest, class 3
4th biggest, class 2
5th biggest, class 1

"""

class ClassificationModel(nn.Module):
    def __init__(self, input_size) -> None:
        super(ClassificationModel, self).__init__()
        self.linear = nn.Linear(input_size, input_size)  # 线性层
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失
        
    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred
        
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)
    return x, 4 - max_index

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    
    model = ClassificationModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model.forward(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.detach().numpy())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    current_directory = os.path.dirname(__file__)
    # Specify the file path
    file_path = os.path.join(current_directory, "model.pt")
    torch.save(model.state_dict(), file_path)
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return 0


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = ClassificationModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # 保存模型
    current_directory = os.path.dirname(__file__)
    # Specify the file path
    file_path = os.path.join(current_directory, "model.pt")
    predict(file_path, test_vec)