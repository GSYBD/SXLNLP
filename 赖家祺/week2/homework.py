# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
实现一个基于交叉熵的三分类任务
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 4)
        self.layer2 = nn.Linear(4, 3)
        self.loss = nn.CrossEntropyLoss()  # 内部已做了softmax处理

        self.activation = torch.softmax

    def forward(self, x):
        x = self.layer1(x)
        y_p = self.layer2(x)
        return y_p


def build_sample():
    x = np.random.random(3)
    if x.max() == x[0]:
        return x, 0
    elif x.max() == x[1]:
        return x, 1
    elif x.max() == x[2]:
        return x, 2


def build_dataset(sample_num):
    X = []
    Y = []
    for i in range(sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    #      测试数据                 分类结果
    return torch.FloatTensor(X), torch.FloatTensor(Y)


def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model.forward(x)  # 等同于 model.forward(x)
        y_pred = model.activation(y_pred, dim=1)
        for y_p, y_t in zip(y_pred, y):
            if y_p[0] == y_p.max() and int(y_t) == 0:
                correct += 1
            elif y_p[1] == y_p.max() and int(y_t) == 1:
                correct += 1
            elif y_p[2] == y_p.max() and int(y_t) == 2:
                correct += 1
            else:
                wrong += 1
    ptg = correct / (correct + wrong)
    print("预测正确率: %f" % ptg)
    return ptg


def main():
    epoch_nums = 20  # 训练轮数
    batch_size = 5  # 一次训练数
    train_sample = 1000  # 样本总数
    input_size = 3
    learning_rate = 0.09  # 学习率

    model = TorchModel(input_size)  # 建模
    optim = torch.optim.Adam(model.parameters(), learning_rate)  # 优化器

    log = []

    # 创建训练集
    # train_x测试数据 , train_y分类结果
    train_x, train_y = build_dataset(train_sample)
    for epoch in range(epoch_nums):
        model.train()
        watch_loss = []
        for batch_idx in range(train_sample // batch_size):
            x = train_x[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            y_t = train_y[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            y_p = model.forward(x)
            # print("x=", x)
            # print("y_p=", y_p)
            # print("y_t=", y_t)
            # print("y_t flatten=", y_t.flatten().long())
            loss = model.loss(y_p, y_t.flatten().long())
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print(f"第{epoch + 1}轮平均loss={np.mean(watch_loss)}")
        acc = evaluate(model)
        log.append([acc, np.mean(watch_loss)])

    torch.save(model.state_dict(), "mymodel.pt")
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return


def predict(model_path, input_vec):
    input_size = 3
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    # print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
        result = model.activation(result, dim=1)
    for vec, res in zip(input_vec, result):
        idx = res.argmax()
        res = ', '.join(f"{val:.2f}" for val in res)
        print(f"输入: {vec}, 输出: {res}, 类别: {idx}")


if __name__ == '__main__':
    # main()

    # test_vec = [np.random.random(3) for i in range(5)]
    # test_vec = np.array(test_vec)
    # print(torch.from_numpy(test_vec))

    test_vec = [[0.2271, 0.6609, 0.9751],
                [0.2694, 0.4865, 0.4558],
                [0.6889, 0.0602, 0.2585],
                [0.5219, 0.1504, 0.5561],
                [0.1983, 0.6682, 0.4302]]
    predict("mymodel.pt", test_vec)
