import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，判断哪个索引数最大，则为第几类

"""


class MyTorchModel(nn.Module):
    def __init__(self, input_size1):
        super(MyTorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size1, 5)

    def forward(self, x):
        x = self.layer1(x)
        return x  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值最大为第一类，以此类推
def build_sample():
    x = np.random.random(5)
    if max(x) == x[0]:
        return x, 0
    elif max(x) == x[1]:
        return x, 1
    elif max(x) == x[2]:
        return x, 2
    elif max(x) == x[3]:
        return x, 3
    else:
        return x, 4


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 1  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = MyTorchModel(input_size)  # 5 * 5 的 线性层
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
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
            y_pred = model(x)  # 计算预测值
            loss = loss_fn(y_pred, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        # print('watch_loss',watch_loss)
        avg_loss = np.mean(watch_loss)
        log.append((epoch, avg_loss))
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, avg_loss))
        print(model)
        # 保存模型
    torch.save(model.state_dict(), "model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    main()
