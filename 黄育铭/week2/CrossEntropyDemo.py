import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

'''
规律：x是一个3维向量，比较3个值的大小，然后返回一个3维向量，最大值的那位给1，例：x = [0.2581, 0.6816, 0.1928]  y = [0,1,0]
'''
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 3)  # 线性层
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y).float()  # 预测值和真实值计算损失
        else:
            return y_pred.float()  # 输出预测结果


def build_sample():
    x = np.random.random(3)
    if x[0] > x[1] and x[0] > x[2]:
        return x, [1,0,0]
    elif x[1] > x[0] and x[1] > x[2]:
        return x, [0, 1, 0]
    else:
        return x, [0, 0, 1]


def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    #return np.array(X)
    #return np.array(X), np.array(Y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)

aa, bb = build_dataset(10)

#print(aa, bb)

def main():
    epoch_num = 50
    batch_size = 20
    train_sample = 5000
    input_size = 3
    learning_rate = 0.003  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #print(optim)
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
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    torch.save(model.state_dict(), "model.pt")
    # 画图
    #print(log)
    # plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    # plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    # plt.legend()
    # plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 3
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        if res[0] > res[1] and res[0] > res[2]:
            aa = [1, 0, 0]
        elif res[1] > res[0] and res[1] > res[2]:
            aa = [0, 1, 0]
        else:
            aa = [0, 0, 1]
        print(vec, res, aa)  # 打印结果

if __name__ == "__main__":
    main()
    # test_vec = [[0.1126, 0.2788, 0.4127],
    #             [0.2581, 0.6816, 0.1928],
    #             [0.2008, 0.8228, 0.4746],
    #             [0.2686, 0.8460, 0.8487],
    #             [0.4855, 0.3824, 0.5189],
    #             [0.4536, 0.2801, 0.3123],
    #             [0.6236, 0.1607, 0.9445],
    #             [0.4245, 0.3504, 0.9271],
    #             [0.3075, 0.8860, 0.9345],
    #             [0.5891, 0.3079, 0.3862]]
    # predict("model.pt", test_vec)