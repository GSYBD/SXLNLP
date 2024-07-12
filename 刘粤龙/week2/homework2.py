# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
"""

基于pytorch框架编写模型训练
实现一个基于交叉熵的多分类任务任务
规律：x是一个3维向量，每一种样本有3个类别

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 3)  # 线性层
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.CrossEntropyLoss()  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 3)
        y_pred = self.activation(x)  # (batch_size, 3) -> (batch_size, 3)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个3维向量，如果样本的最大值是第一个值，则认为是第一种类样本，以此类推
def build_sample():
    x = np.random.random(3)
    return x, np.argmax(x)


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    tp = [0, 0, 0]  # 统计每类样本数量
    for i in range(total_sample_num):
        x, y = build_sample()
        tp[y] += 1
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y), tp


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y, tp = build_dataset(test_sample_num)
    print("本次预测集中共有第一类样本%d个，第二类样本%d个，第三类样本%d个" % (tp[0], tp[1], tp[2]))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if np.argmax(y_p) == y_t:
                correct += 1  # 样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 3  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y, tp = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
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


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 3
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for index, vec in enumerate(input_vec):
        print("第 %d 个输入样本：%s" % (index, vec))  # 打印结果
    #
    for index, i in enumerate(result):
        predicted_class = torch.argmax(i).item()
        probability = i[predicted_class]  # 获取对应类别的概率值
        print("第 %d 个输入样本, 预测类别为：%d, 概率值：%f" % (index, predicted_class, probability))  # 打印结果

if __name__ == "__main__":
    main()
    test_vec = [[0.07889086, 0.15229675, 0.31082123],
                [0.94963533, 0.5524256, 0.95758807],
                [0.78797868, 0.67482528, 0.13625847],
                [0.79349776, 0.59416669, 0.92579291]]
    predict("model.pt", test_vec)
