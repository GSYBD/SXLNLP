#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
基于torch框架
规律：x为5维向量，如果第1个数 > 第5个数字，为正样本，反之为负样本
判断：正、负样本，实际是一个二分类的任务
"""


# 定义模型结构
class MultiClass(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.loss = nn.functional.cross_entropy  # loss函数为交叉熵

    # 输入真实标签（即训练数据中有y值），返回loss值。无真实标签，返回预测值。
    def forward(self, x, y=None):
        yp = self.linear(x)
        if y is not None:
            return self.loss(yp, y)
        else:
            return yp


def build_5v():
    """
    用于生成5维向量
    :return: v5 为训练数据，数字0、1为真实值
    return: (array([0.52810604, 0.78161287, 0.17738479, 0.88700755, 0.14389649]), 1)
    return: (array([0.30548794, 0.71524054, 0.00606467, 0.32885616, 0.94656252]), 0)
    """
    v5 = np.random.random(5)  # 生成一个5维向量：[0.92484474 0.65475379 0.29321597 0.4388626  0.53941327]
    max_index = np.argmax(v5)
    return v5, max_index


def traindata(num):
    """
    用于生成指定 num 数量的训练数据
    数据输出格式如下：
    (
        tensor([[0.8374, 0.1586, 0.4920, 0.1052, 0.6836],
                [0.6528, 0.1058, 0.3741, 0.3993, 0.8427]]),
        tensor([[1.],
                [0.]])
    )
    """
    X = []
    Y = []
    for i in range(num):
        x, y = build_5v()
        X.append(x)
        Y.append(y)
    # torch.LongTensor(Y), 必须为Long类型，计算需要标量
    return torch.FloatTensor(X), torch.LongTensor(Y)


def testmodel(model):
    """
    :param model: 测试模型model
    :return: 返回该模型的准确率
    """
    model.eval()  # 告诉模型，现在开始测试模式
    t_data_num = 100  # 测试样本数量
    x, y = traindata(t_data_num)  # 生成测试训练集，工作中实际上为读取数据
    correct, wrong = 0, 0
    with torch.no_grad():  # torch中上下文管理器，告诉torch不需要计算梯度，目的是为了减小计算资源消耗
        ypred = model(x)  # 在模型定义中，如果不输入y就会输出预测值，这里只传入x，得到预测值
        for yp, yt in zip(ypred, y):  # 使用zip函数将yp（loss值）和y进行一一对应，[(yp1, y1), (yp2, y2)...]，是一个可迭代对象
            if torch.argmax(yp) == int(yt):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数为：{}，准确率为：{}".format(correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练多少轮
    data_num = 5000  # 每轮训练总共训练样本总数
    batch_size = 20  # 每次训练样本个数
    input_size = 5  # 向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = MultiClass(input_size)
    # 选择优化器：adam，优化器的主要目的是调整模型的参数（如权重和偏置），以最小化损失函数，从而提高模型的性能
    # model.parameters()：是 PyTorch 中 nn.Module 类的一个方法（迭代器），是模型中需要学习和更新的权重和偏置项，参数通常包括每一层的权重和偏置项
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []  # 创建该列表的目的是为了画图使用，不画图可删除该变量
    # 获取训练集，正常情况下读取训练集数据
    train_x, train_y = traindata(data_num)
    #  训练过程
    for epoch in range(epoch_num):
        model.train()  # 告诉模型可以训练
        watch_loss = []  # 创建watch_loss的目的是为了画图观察loss使用，不画图可删除该变量
        for bi in range(data_num // batch_size):  # 将数据按20条切分为一批
            x = train_x[bi * batch_size: (bi + 1) * batch_size]  # 列表的切片操作，[0: 20]，一次获取20条样本
            y = train_y[bi * batch_size: (bi + 1) * batch_size]  # 同上
            loss = model(x, y)  # 计算损失，等同于model.forward(x,y)，forward等于模型定义的foward函数。model(x,y)是简写
            loss.backward()  # 计算梯度，计算当前损失loss的梯度的函数，反向传播算法的一部分，用于计算损失函数相对于模型参数的梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 更新权重之后，将梯度归零
            watch_loss.append(loss.item())  # 将差异值添加到列表中，用于画图展示变化
        print("第{}轮，平均loss为：{}".format(epoch + 1, np.mean(watch_loss)))
        # 评估本轮训练模型的准确率
        acc = testmodel(model)
        log.append([acc, float(np.mean(watch_loss))])
    # 训练完成之后，将模型保存下来，model.state_dict()模型的权重参数
    torch.save(model.state_dict(), "MultiClass.pt")
    # 通过画图，展示出来变化
    # print(log)
    plt.plot(range(len(log)), [i[0] for i in log], label='acc')  # acc线，准确率
    plt.plot(range(len(log)), [i[1] for i in log], label='loss')  # loss线
    plt.legend()
    plt.show()
    return


def predict(model_path, input_vec):
    """
    用于测试训练好的模型的准确率
    :param model_path: 模型路径
    :param input_vec: 输入的测试数据
    :return:
    """
    input_size = 5
    model = MultiClass(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print("模型权重------\n", model.state_dict(), "\n------模型权重")  # 输出模型权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：{}, 预测类别：{}".format(vec, torch.argmax(res)))  # 打印结果



if __name__ == '__main__':
    # 模型真正训练的步骤
    # main()
    # 使用训练好的模型做预测
    test_vec = [[0.58973582, 0.15008354, 0.293453  , 0.38442002, 0.48170333],
                [0.53959869, 0.49386259, 0.3060606 , 0.80275197, 0.49112025],
                [0.48992686, 0.36135758, 0.57196666, 0.04939117, 0.63128775],
                [0.60015942, 0.63575095, 0.66289441, 0.81388397, 0.73560092],
                [0.39799916, 0.80885392, 0.34542767, 0.45826153, 0.00901708]]
    predict("MultiClass.pt", test_vec)
"""
通过调整学习率：0.01，多次训练模型
测试模型：
给定测试数据：
    (array([0.58973582, 0.15008354, 0.293453  , 0.38442002, 0.48170333]), 0)
    (array([0.53959869, 0.49386259, 0.3060606 , 0.80275197, 0.49112025]), 3)
    (array([0.48992686, 0.36135758, 0.57196666, 0.04939117, 0.63128775]), 4)
    (array([0.60015942, 0.63575095, 0.66289441, 0.81388397, 0.73560092]), 3)
    (array([0.39799916, 0.80885392, 0.34542767, 0.45826153, 0.00901708]), 1)
预测结果：
    输入：[0.58973582, 0.15008354, 0.293453, 0.38442002, 0.48170333],   预测类别：0
    输入：[0.53959869, 0.49386259, 0.3060606, 0.80275197, 0.49112025],  预测类别：3
    输入：[0.48992686, 0.36135758, 0.57196666, 0.04939117, 0.63128775], 预测类别：4
    输入：[0.60015942, 0.63575095, 0.66289441, 0.81388397, 0.73560092], 预测类别：3
    输入：[0.39799916, 0.80885392, 0.34542767, 0.45826153, 0.00901708], 预测类别：1  
"""
