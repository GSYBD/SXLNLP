import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
"""
构建一个x三维向量，在五个维度找出其中最大数的维度，并分成三类
"""
# 生成随机的w和d
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 3)
        self.activation = torch.sigmoid
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred

# print(type(x))  查看数据类型

# 随机生成3维向量，找到最大值是第几个，并输出分类类型
def build_sample():
    x = np.random.random((3))   # 定义一个张量（（x,y,z））,表示x个y*z向量 x类型：numpy.ndarray
    max = x[0]
    n = 1
    for i in range(x.size):     #x.size表示元素个数
        if max < x[i]:
            max = x[i]
            n = i+1
    return x, n

# 输入样本数 生成一批样本 输出样本及样本类型
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        n = [0, 0, 0]
        n[y-1] = 1
        Y.append(n)
    return torch.FloatTensor(X), torch.FloatTensor(Y)   #torch.Tensor类型

# 将生成的样本进行样本分类
def classify(test_sample_y):
    n = test_sample_y.flatten()      # 将任意维度张量转化为一维张量
    a = 1
    if n[0].item() > n[1].item() and n[0].item() > n[2].item():
        return 1
    elif n[1].item() > n[0].item() and n[1].item() > n[2].item():
        return 2
    else:
        return 3



# 测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model.forward(x)  # 模型预测   =model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            p = classify(y_p)
            t = classify(y_t)

            if p == t:
                correct += 1  # 负样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)



epoch_num = 20  # 训练轮数
batch_size = 20  # 每次训练样本个数
train_sample = 500  # 每轮训练总共训练的样本总数
input_size = 3  # 输入向量维度
learning_rate = 0.4  # 学习率
# 建立模型
model = TorchModel(input_size)
# 选择优化器     Adam优化器名称 做权重更新
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
log = []
train_x, train_y = build_dataset(train_sample)
# 训练过程
for epoch in range(epoch_num):
    model.train()
    watch_loss = []
    for batch_index in range(train_sample):
        # x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
        # y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
        x = train_x[batch_index]
        y = train_y[batch_index]
        # print(x)
        # print("y",y)
        loss = model.forward(x, y)
        # print("y_pred",model.forward(x))
        # print(loss)
        loss.backward()
        optim.step()
        optim.zero_grad()
        watch_loss.append(loss.item())
        # print(model.state_dict())
        # # print(x)
        # print(y)
        # print(model.forward(x))
#
    print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    # print(model.state_dict())
    acc = evaluate(model)  # 测试本轮模型结果
    # print("acc",acc)
    log.append([acc, float(np.mean(watch_loss))])

# 保存模型
torch.save(model.state_dict(), "model.pt")
# 画图
print(log)
plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
plt.legend()
plt.show()



