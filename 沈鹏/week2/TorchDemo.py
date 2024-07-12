import numpy.random
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本
"""

# 创建一个TorchModel继承nn.Module 并实现init，forward方法
class TorchModel(nn.Module):
    def __init__(self, hidden_size):
        super(TorchModel, self).__init__()
        # 构建网络层，线性组件，激活函数组件，损失函数组件
        # 线性层 一层线性
        self.linear = nn.Linear(hidden_size, 1)
        # 激活函数
        self.activation = torch.sigmoid
        # 损失函数 使用交叉熵
        self.loss = nn.functional.mse_loss

    # 当输入真实标签，返回loss值，无真实标签，返回预测值
    def forward(self, x, y=None):
        # 输出
        x = self.linear(x)
        # activation是什么意思？
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果

# 生成一个样本，样本的生成方法，代表我们要学习的规律
# 随机生成一个5维向量，如果第1个值大于第5个值，认为是正样本，反之为负样本
def build_sample():
    x = numpy.random.random(5)
    if x[0] > x[4]:
        return x, 1
    else:
        return x, 0

# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        # 此处做标记
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    # 这个方法意思不知道
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本轮预测集中共有%d个正样本，%d个负样本"%(sum(y), test_sample_num - sum(y)))
    correct, wrong = 0,0
    with torch.no_grad():
        y_pred = model(x)# 模型预测
        for y_p,y_t in zip(y_pred, y):# 与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1#负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1 #正样本判断正确
            else:
                wrong += 1

    print("正确预测个数：%d,正确率%f" %(correct, correct/(correct + wrong)))
    return correct/(correct + wrong)

def main():
    # 配置参数
    epoch_num = 20   # 训练轮数
    batch_size = 20  # 每次训练的样本个数
    train_sample = 5000  # 每轮训练总共训练的样本数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率

    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器 优化器是什么？有什么作用
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
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()     # 计算梯度
            optim.step()    # 更新权重
            optim.zero_grad()  # 梯度归0
            watch_loss.append(loss.item())
        print("\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
# 保存模型
    torch.save(model.state_dict(), "model.pt")
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

if __name__ == '__main__':
    main()