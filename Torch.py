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
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本
1.五分类任务：第一个数最大 为1分类
          第二个数最大为2分类
          ....
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        # self.loss = nn.CrossEntropyLoss()  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x):
        return self.linear(x)  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    x = np.random.random(5)
    max_idx = np.argmax(x)  # 找到最大值的索引
    y = max_idx  # 索引作为类别标签（0到4）
    return torch.from_numpy(x).float(), y


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    label_counts = [0] * 5  # 初始化标签计数列表
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
        label_counts[y] += 1
    X = torch.stack(X)
    return torch.FloatTensor(X), torch.LongTensor(Y), label_counts

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y, label_counts = build_dataset(test_sample_num)
    print("本次数据集中一分类个数：%d\n本次数据集中二分类个数：%d\n本次数据集中三分类个数：%d\n本次数据集中四分类个数：%d\n本次数据集中五分类个数：%d\n" \
          % (label_counts[0], label_counts[1], label_counts[2], label_counts[3], label_counts[4]))
    correct, wrong = 0, 0
    with torch.no_grad(): # 不计算梯度
        y_pred = model(x)  # 模型预测
        # print(y_pred)
        y1 = []
        counts_pred = [0] * 5  # 初始化标签计数列表
        for counts in y_pred:  # 对预测数进行标签统计
            max_idx = torch.argmax(counts)
            y1.append(max_idx)
        y1 = torch.LongTensor(y1)
        for j in y1:
            counts_pred[j] += 1
        print("真实值:", y1)
        print("预测值:", y)
        # 比较两个张量，结果是一个布尔类型的张量
        equal_mask = y == y1

        # 计算相等值的数量
        correct = equal_mask.sum().item()
    print("本次预测集中一分类个数：%d\n本次预测集中二分类个数：%d\n本次预测集中三分类个数：%d\n本次预测集中四分类个数：%d\n本次预测集中五分类个数：%d\n" \
          % (counts_pred[0], counts_pred[1], counts_pred[2], counts_pred[3], counts_pred[4]))
    print(label_counts)
    print(counts_pred)
    print(correct, y.size(0))
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / y.size(0)))
    return correct, correct / y.size(0)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.002  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y, label_counts = build_dataset(train_sample)

    cross = nn.CrossEntropyLoss()
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):    
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            y_pred = model(x)
            # print(y_pred)
            # print(y)
            loss = cross(y_pred, y)  # 计算loss
            # print(loss)
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
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    y1 = []
    counts_pred = [0] * 5  # 初始化标签计数列表
    for counts in result:  # 对预测数进行标签统计
        max_idx = torch.argmax(counts)
        y1.append(max_idx)
    for j in y1:
        counts_pred[j] += 1
    print("本次预测集中一分类个数：%d\n本次预测集中二分类个数：%d\n本次预测集中三分类个数：%d\n本次预测集中四分类个数：%d\n本次预测集中五分类个数：%d\n" \
          % (counts_pred[0], counts_pred[1], counts_pred[2], counts_pred[3], counts_pred[4]))
    y1 = torch.LongTensor(y1)
    print(y1)

if __name__ == "__main__":
    # main()
    test_vec = [[0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.79349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
    predict("model.pt", test_vec)

