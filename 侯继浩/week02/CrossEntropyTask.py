# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
"""

基于pytorch框架编写模型训练
实现一个基于交叉熵的多分类任务
规律：x是一个5维向量，其中每个元素是0~9之间的整数
如果第2个数+第4个数小于6，则为分类1,
如果第2个数+第4个数大于等于6且小于12，则为分类2,
如果第2个数+第4个数大于12，则为分类3

"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 3)  # 线性层
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 随机生成目标样本
def build_sample():
    x = np.random.randint(0, 9, 5)
    if x[1] + x[3] < 6:
        return x, [1, 0, 0]
    elif 6 <= x[1] + x[3] < 12:
        return x, [0, 1, 0]
    else:
        return x, [0, 0, 1]

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))

def get_class_label(x):
    labels = [1, 2, 3]
    return labels[int(x.numpy().argmax())]

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    yRes = []
    for y1 in y:
        yRes.append(int(get_class_label(y1)))
    yCounter = Counter(yRes)
    print("本次预测集中共有%d个1分类，%d个2分类，%d个3分类" % (yCounter[1], yCounter[2], yCounter[3]))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if int(get_class_label(y_p)) == int(get_class_label(y_t)):
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


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
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            if len(x) > 0:
                loss = model(x, y)  # 计算loss
                loss.backward()  # 计算梯度
                optim.step()  # 更新权重
                optim.zero_grad()  # 梯度归零
                watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.task")
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
        print("输入：%s, 预测类别：%d" % (vec, get_class_label(res)))  # 打印结果


if __name__ == "__main__":
    main()
    # test_vec = [[1, 2, 1, 9, 1],
    #             [5, 5, 2, 0, 9],
    #             [1, 7, 5, 1, 1],
    #             [2, 9, 1, 7, 1]]
    # predict("model.task", test_vec)
