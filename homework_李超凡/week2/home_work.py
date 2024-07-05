# encoding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
随机生成100以内包含10个元素的以为向量
若0,1位置的元素和最大，输出类别0
若2,3位置的元素和最大，输出类别1
若4,5位置的元素和最大，输出类别2
若6,7位置的元素和最大，输出类别3
若8,9位置的元素和最大，输出类别4
"""


# 构造数据集
def construct_dataset(counts):
    x = []
    y = []
    for i in range(counts):
        x_ = np.random.randint(100, size=10)
        y_ = np.argmax(np.sum(x_.reshape(-1, 2), axis=1))
        x.append(x_)
        y.append(y_)
    return torch.FloatTensor(x), torch.LongTensor(y)


# 搭建模型
class MultiClassficationModel(nn.Module):
    def __init__(self, input_size):
        super(MultiClassficationModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y = None):
        y_predict = self.linear(x)
        if y is not None:
            return self.loss(y_predict, y)
        else:
            return y_predict


# 模型准确率评价
def evaluate(model, test_sample):
    model.eval()
    x, y = construct_dataset(test_sample)
    correct, wrong = 0, 0
    with torch.no_grad():  # 不计算梯度，降低内存和计算量开支
        y_predict = model(x)
        for y_p, y_t in zip(y_predict, y):
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 模型训练
def train(epoch_num, batch_size, train_sample, input_size, test_sample, learning_rate=0.001):
    # 实例化模型
    model = MultiClassficationModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 构造数据集
    train_x, train_y = construct_dataset(train_sample)
    # 开始训练
    train_log = []  # 训练日志
    for epoch in range(epoch_num):
        model.train()
        loss_ls = []
        for batch_idx in range(train_sample // batch_size):
            x = train_x[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            y = train_y[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            loss_ls.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(loss_ls)))
        accuracy = evaluate(model, test_sample)
        train_log.append([accuracy, np.round(float(np.mean(loss_ls)), 5)])

    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    print(train_log)
    # 可视化训练过程
    plt.plot(range(len(train_log)), [l[0] for l in train_log], label="acc")  # 画acc曲线
    plt.plot(range(len(train_log)), [l[1] for l in train_log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

    return


# 使用训练好的模型进行预测
def predict(model_path, input_size, validation_sample):
    model = MultiClassficationModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    validation_x, validation_y = construct_dataset(validation_sample)
    with torch.no_grad():
        result = model.forward(validation_x)  # 模型预测
        for vec, res in zip(validation_x, result):
            print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res))


if __name__ == "__main__":
    epoch_num=20
    batch_size=20
    train_sample=10000
    input_size=10
    test_sample=2000
    learning_rate = 0.001
    validation_sample = 20
    train(epoch_num, batch_size, train_sample, input_size, test_sample, learning_rate)
    predict("model.pt", 10,validation_sample)
