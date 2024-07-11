# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，找出向量中的最大值

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 输入层到隐藏层的线性映射
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # 通过输出层
        y_pred = self.log_softmax(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    x = np.random.random(5)
    if x[0] == max(x):
        return x, 0
    elif x[1] == max(x):
        return x, 1
    elif x[2] == max(x):
        return x, 2
    elif x[3] == max(x):
        return x, 3
    elif x[4] == max(x):
        return x, 4


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    y_zero, y_one, y_two, y_three, y_four = 0, 0, 0, 0, 0
    for value in y:
        if value == 0:
            y_zero += 1
        elif value == 1:
            y_one += 1
        elif value == 2:
            y_two += 1
        elif value == 3:
            y_three += 1
        elif value == 4:
            y_four += 1
    print("本次预测集中共有%d个一类样本，%d个二类样本，%d个三类样本，%d个四类样本，%d个五类样本" % (y_zero, y_one, y_two, y_three, y_four))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        # 确保y_pred_labels是预测的最大类别索引
        y_pred_labels = torch.argmax(y_pred, dim=1)
        # 计算正确预测的数量
        correct = (y_pred_labels == y).sum().item()
        accuracy = correct / test_sample_num
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return correct / test_sample_num


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
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
    torch.save(model.state_dict(), "model.pt")
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
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        inputs = torch.tensor(input_vec, dtype=torch.float32)  # 确保输入数据类型正确
        result = model(inputs)
        predicted_labels = result.argmax(dim=1)  # 获取预测的最大类别索引
    for vec, pred_label in zip(input_vec, predicted_labels.tolist()):  # 使用tolist()转换为列表
        print("输入：%s, 预测类别：%d" % (vec, pred_label + 1))

if __name__ == "__main__":
    # main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.98920843],
                [0.94963533,0.5524256,0.95758807,0.95760434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.94675372,0.19871392],
                [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.pt", test_vec)
