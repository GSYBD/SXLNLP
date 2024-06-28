# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数是最大值，则该向量为1类向量，如果第2个数为最大值，则该向量为2类向量，以此类推

"""

"""
构建以神经网络模型为参数的自定义模型
"""
class TorchModel(nn.Module):
    """
    创建构造函数，输入参数为向量维度
    """
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        # self.linear = nn.Linear(input_size, 1)  # 线性层，输入维度为input_size，输出维度为1
        self.layer1 = nn.Linear(input_size, 10) #w：3 * 5
        self.layer2 = nn.Linear(10, 5) # 5 * 2
        # self.activation = torch.sigmoid  # sigmoid归一化函数，希望最终输出值在0-1之间，因此要用sigmoid函数对结果进行转换
        self.loss = nn.functional.mse_loss  # loss函数采用均方差损失
        # self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.layer1(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.layer2(x)
        print(y)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
#生成随机样本，如果第1个数是最大值，则该向量为1类向量，如果第2个数为最大值，则该向量为2类向量，以此类推
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)
    if max_index == 0:
        return x, 0
    elif max_index == 1:
        return x, 1
    elif max_index == 2:
        return x, 2
    elif max_index == 3:
        return x, 3
    else:
        return x, 4


def getMaxIndex(arr):
    for index,value in enumerate(arr):
        if value == max(arr):
            return index


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 等价于model.forward(x) 得到预测值
        # print(y_pred)
        for y_p, y_t in zip(y_pred, y): # 与真实标签进行对比
            if getMaxIndex(y_p)== int(y_t):
                correct += 1  # 负样本判断正确
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数，实际工作中训练数据是有限的
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器,此处优化器选择自带的adam优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()  ##把模型定义在训练的阶段
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss，因为y不为none，输出结果为Loss，即model.forward()
            loss.backward()  # 计算梯度，框架自带用法
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零 主要目的是计算下一batch，当前Batch归零处理
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
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.pt", test_vec)
