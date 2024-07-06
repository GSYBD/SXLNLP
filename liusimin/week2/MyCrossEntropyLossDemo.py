# coding:utf8

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
随机生成N维向量,随机范围 [1,maxrandom=10*N!],构造N+1级别均匀分布,满足以下规则:cal_label
例如对于4维向量,随机构造 [1,1200]以内4维向量 [1,1200]
第一个数小于240 第0类，否则看第二个数
第2个数小于300 第1类，否则看第3个数
第3个数小于400 第2类，否则看第4个数
第4个数小于600 第3类，否则是第4类 
"""

N = 4
maxrandom = 10 * math.factorial(N)


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个N维向量，根据每个向量中最大的标量同一下标构建Y
def build_sample():
    x = np.random.randint(1, maxrandom, N)
    return x, cal_label(x)


def cal_label(x):
    label = 0
    for i in range(0, N):
        imin = maxrandom / (N - i + 1)
        if x[i] > imin:
            label = label + 1
        else:
            return label
    return label


def build_dataset(total_sample_num):
    """
    生成一批样本
    :param total_sample_num:
    :return:
    """
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return TensorDataset(torch.FloatTensor(np.array(X)), torch.LongTensor(Y))


def evaluate(model):
    """
    模型测试
    :param model:
    :return:
    """
    model.eval()
    test_sample_num = 100
    dataset = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(dataset.tensors[0])  # 模型预测
        for y_p, y_t in zip(y_pred, dataset.tensors[1]):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


class MultiClassficationModel(nn.Module):
    def __init__(self, input_size):
        super(MultiClassficationModel, self).__init__()
        self.linear = nn.Linear(input_size, N + 1)  # 线性层 一共有N+1种分类
        self.activation = nn.Softmax(dim=1)
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)
        y_pred = self.activation(y_pred)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


def main():
    # 配置参数
    epoch_num = 40  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 100000  # 每轮训练总共训练的样本总数
    input_size = N  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = MultiClassficationModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    dataset = build_dataset(train_sample)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 训练过程
    for epoch in range(epoch_num):
        watch_loss = []
        for inputs, labels in dataloader:
            optim.zero_grad()  # 梯度归零
            loss = model(inputs, labels)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重

            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    # torch.save(model.state_dict(), "model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    main()
