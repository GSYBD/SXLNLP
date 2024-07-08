# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from torchsummary import summary
"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，最大的值对应的索引即为类别

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 5)
        self.activation = nn.Softmax(dim=1)
        self.criterion = nn.NLLLoss()

    def forward(self, x, y=None):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        y_pred = self.activation(x)

        if y is not None:
            y = y.flatten()
            return self.criterion(torch.log(y_pred), y)
        else:
            return y_pred


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大的值对应的索引即为类别
def build_sample():
    x = np.random.random(5)
    return x, np.argmax(x)


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.tensor(Y, dtype=torch.long)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)


    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        predicted_classes = torch.argmax(y_pred, dim=1)  # 获取最高概率的类别索引
        # print(predicted_classes)
        y_true = y.flatten()
        # print('x:', x)
        # print(y_true, y_pred)
        # print('y_pred:', predicted_classes)
        correct_predictions = (predicted_classes == y_true).sum().item()
    # print("预测类别：%s, 真实类别：%s" % (predicted_classes.numpy(), y_true))
    print("正确率：%f" % (correct_predictions / y_pred.shape[0] ))
    return correct_predictions


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    hidden_size = 20  # 隐藏层向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size, hidden_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model_5class.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    hidden_size = 20
    model = TorchModel(input_size, hidden_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    predicted_classes = torch.argmax(result, dim=1)  # 获取最高概率的类别索引
    return predicted_classes.numpy()


if __name__ == "__main__":
    main()
    # print(build_sample())
    # model = TorchModel(5, 10)
    # summary(model)
    # 对应的类别是【2， 2， 0， 2】
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    print(predict("model_5class.pt", test_vec))

