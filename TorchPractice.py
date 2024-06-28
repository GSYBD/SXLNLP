import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt


"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，向量中最大的数

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵

    # 当输入真实标签 返回loss 值 ；无真实标签 返回预测值

    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, input_size) ->(batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y) # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测值结果


# 生成一个随机样本，样本的生成方法，代表了我们要学习的规律
# 随机生成一个五维向量，如果第一个值大于第五个值，则认为是正样本，反之为负样本

def bulid_sample():
    x = np.random.random(5)
    target_one = np.zeros(x.shape)
    i = [index for index, item in enumerate(x) if item == x.max()]
    target_one[i] = 1
    return x, target_one


# 随机生成一批样本
# 正负样本均匀生成
def bulid_dataset(total_sample_num):
    X = []
    Y = []


    for i in range(total_sample_num):
        x, y= bulid_sample()
        X.append(x)
        Y.append(y)

    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率


def evaluate(model):
    model.eval()
    test_sample_sum = 100
    x, y = bulid_dataset(test_sample_sum)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            y_p_max = [index for index, item in enumerate(y_p) if item == y_p.max()]
            y_t_max = [index for index, item in enumerate(y_t) if item == y_t.max()]
            print(y_p_max,y_t_max)
            if y_p_max == y_t_max :
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d，正确率 ：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练的轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入的向量纬度
    learning_rate = 0.001  # 学习率

    # 建立模型
    model = TorchModel(input_size)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = bulid_dataset(train_sample)

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
            optim.zero_grad  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return

if __name__ == "__main__":
   main()



