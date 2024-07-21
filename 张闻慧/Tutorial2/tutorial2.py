# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律的多分类任务 - 用交叉熵实现
规律：x是一个10维向量，所有值在0到1之间
如果前五维的平均值大于后五维的平均值，并且至少有三个值大于0.5，则为类别1
如果前五维的平均值小于后五维的平均值，并且至少有三个值小于0.5，则为类别2
其他情况为类别0
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 4)  # 线性层
        self.linear2 = nn.Linear(4, 3)  # 线性层
        self.relu = torch.relu
        self.softmax = nn.Softmax(dim=1)  # softmax归一化函数
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失


    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear1(x)  # (batch_size, input_size) -> (batch_size, 4)
        x = self.relu(x)  # (batch_size, 4) -> (batch_size, 4)
        x2 = self.linear2(x) # (batch_size, 4) -> (batch_size, 3)
        y_pred = self.softmax(x)  # (batch_size, 3)

        if y is not None:
            return self.loss(x2, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
def build_sample():
    x = np.random.random(10)
    if np.mean(x[:5]) > np.mean(x[5:]) and np.sum(x[:5] > 0.5) >= 3:
        return x, 1
    elif np.mean(x[:5]) < np.mean(x[5:]) and np.sum(x[5:] < 0.5) >= 3:
        return x, 2
    else:
        return x, 0

# 随机生成一批样本
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
    print("本次预测集中共有%d个类别1，%d个类别2，%d个类别0" % (np.sum(y==1), np.sum(y==2), np.sum(y==0)))
    correct, total = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        _, predicted = torch.max(y_pred, 1)  # 获取预测类别
        correct += (predicted == y).sum().item()  # 正确预测的数量
        total += y.size(0)  # 总样本数量
    print("准确率：%f" % (correct / total))
    return correct / total


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 30  # 每次训练样本个数
    train_sample = 6000  # 每轮训练总共训练的样本总数
    input_size = 10  # 输入向量维度
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
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 10
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())

    model.eval()  # 测试模式
    test_p = []
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果
        test_p.append([round(float(res))])
    return test_p



def build_test_dataset(total_test_num):
    X = []
    Y = []
    for i in range(total_test_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return X,Y


if __name__ == "__main__":
    main()
    test_vec, test_gt = build_test_dataset(50)
    predict_label = predict("model.pth", test_vec)
    accuracy = np.sum(np.array(predict_label) == np.array(test_gt))/ 50
    print("The accuracy out of 50 predicted samples is", accuracy)



