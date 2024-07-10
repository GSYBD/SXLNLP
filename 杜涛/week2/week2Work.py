"""
week2作业
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

"""
模型训练
找出1*5向量中的最大值实现五分类任务
"""


# 模型
class TorchModel(nn.Module):

    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        # self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = F.cross_entropy  # loss函数采用均方差

    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = x
        if y is not None:
            return self.loss(y_pred, y.long().view(-1))  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x = np.random.random(5)
        y = np.argmax(x)
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)


def eavaluate(model):
    x, y = build_dataset(100)
    y_pred = model(x)
    correct = 0
    # 调用模型
    with torch.no_grad():
        # 测试
        model.eval()
        for y_pred, y in zip(y_pred, y):
            if torch.argmax(y_pred) == y:
                correct += 1
        # 输出正确的个数和正确率
        print(f"correct的个数: {correct}")
        print(f"正确率: {correct / 100}")


def main():
    # 1.设置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入的向量维度
    learning_rate = 0.001  # 学习率

    # 2.建立模型
    model = TorchModel(input_size);

    # 3.选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 记录日志
    Log = []

    # 4.创建训练集，得到预测值
    train_x, train_y = build_dataset(train_sample)
    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    dataset_batch = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 5.训练过程
    for i in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in dataset_batch:
            x, y = batch_index
            loss = model(x, y);  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
            acc=eavaluate(model)  # 评估函数
            # print(f"epoch: {epoch}, loss: {sum(batch_loss) / len(batch_loss)}")
            Log.append([acc, float(np.mean(watch_loss))])

    # 6.保存模型
    torch.save(model.state_dict(), "model.pt")
    # 7.画图
    print(Log)
    plt.plot(range(len(Log)), [l[0] for l in Log], label="acc")  # 画acc曲线
    plt.plot(range(len(Log)), [l[1] for l in Log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # main()

    # 使用训练好的模型进行预测

    # 加载模型
    model = TorchModel(5)
    model.load_state_dict(torch.load('model.pt'))

    # 测试
    model.eval()
    x = torch.randn(5, 5)
    y = model(x)
    print("x.model:",x)
    print("y.model:",y)
    print(torch.argmax(y))

