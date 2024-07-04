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
规律：x是一个3维向量，如果第1个数最大，则为Ⅰ类样本；如果第二个数最大，则为Ⅱ类样本，如果第三个数最大，则为Ⅲ类样本

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 3)
        # self.linear2 = nn.Linear(hiden_size1, hiden_size2)# 线性层
        self.activation = torch.softmax  # 激活函数sigmoid
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)
        # x = self.linear2(x)
        y_pred = self.activation(x,0)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个3维向量，根据最大值进行分类
def build_sample():
    x = np.random.rand(3)
    if(x[0]>=x[1] and x[0]>=x[2]):
        return x, [1, 0, 0]
    elif(x[1]>=x[0] and x[1]>=x[2]):
        return x, [0, 1, 0]
    elif (x[2]>=x[0] and x[2]>=x[1]):
        return x, [0, 0, 1]




# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    sum_y = sum(y)
    # print(sum_y[0])
    print("本次预测集中共有%d个Ⅰ类样本，%d个Ⅱ类样本，%d个Ⅲ类样本" % (sum_y[0],sum_y[1],sum_y[2]))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if (y_p[0] >= y_p[1] and y_p[0] >= y_p[2]) and y_t.equal(torch.tensor([1., 0., 0.])):
                correct += 1  # Ⅰ类样本判断正确
            elif (y_p[1] >= y_p[0] and y_p[1] >= y_p[2]) and y_t.equal(torch.tensor([0., 1., 0.])):
                correct += 1  # Ⅱ类样本判断正确
            elif (y_p[2] >= y_p[0] and y_p[2] >= y_p[1]) and y_t.equal(torch.tensor([0., 0., 1.])):
                correct += 1  # Ⅲ类样本判断正确
            else:
                wrong += 1
            # print(y_p,y)
    # print(y_pred,y)
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct+wrong)))
    return correct / (correct+wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 3  # 输入向量维度
    # hiden_size1 = 4
    # hiden_size2 = 3
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
        # evaluate(model)
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    # 画图
    # print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 3  # 输入向量维度
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        if(res[0]>=res[1] and res[0]>=res[2]):
            print("输入：%s, 预测类别：Ⅰ类，概率：%f" % (vec, res[0]/(res[0]+res[1]+res[2])))
        elif(res[1]>=res[0] and res[1]>=res[2]):
            print("输入：%s, 预测类别：Ⅱ类，概率：%f" % (vec, res[1] / (res[0] + res[1] + res[2])))
        else:
            print("输入：%s, 预测类别：Ⅲ类，概率：%f" % (vec, res[2] / (res[0] + res[1] + res[2])))
        # print("输入：%s, 预测类别：%s" % (vec,res))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.15229675,0.18920843],
                [0.94963533,0.95758807,0.95520434],
                [0.78797868,0.67482528,0.19871392],
                [0.59416669,0.41567412,0.1358894]]
    predict("model.pt", test_vec)
