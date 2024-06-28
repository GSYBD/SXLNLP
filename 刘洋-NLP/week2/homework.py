import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
#作业内容:实现一个基于交叉熵的多分类任务
# 实现一个五分类的机器学习任务
# 任务要求:
# 第一个数+第二个数>=第四个数+第五个数，则为类别1,反之类别为0

# 定义神经网络模型
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # sigmoid归一化函数
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.CrossEntropyLoss()

#定义反向传播函数(数据传播定义)
    def forward(self, x, y = None):
        x = self.linear(x)
        y_pred = self.activation(x)
        # 当输入真实标签，返回loss值；无真实标签，返回预测值
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


# 生成一个样本：随机生成一个5维向量，如果第一个数+第二个数>=第四个数+第五个数，则为类别1,反之类别为0
def build_sample():
    x = np.random.random(5)
    if x[0] + x[1] >= x[3] + x[4]:
        return x, [1,0,0,0,0]
    else:
        return x, [0,0,0,0,1]

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        # print(y_pred,y)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if y_p[0] + y_p[1]>= y_p[3] + y_p[4] and y_t[0] == 1:
                correct += 1
            elif y_p[0] + y_p[1]< y_p[3] + y_p[4] and y_t[4] == 1:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct,correct + wrong
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
    # print(train_x, train_y)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  等级于model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss)}")
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
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
        idx = np.unravel_index(res.argmax(), res.shape)
        result = np.zeros_like(res)
        result.flat[np.ravel_multi_index(idx, res.shape)] = 1
        print(vec, result)  # 打印结果




if __name__ == '__main__':
    # main()
    test_vec = [[0.9619, 0.7825, 0.3299, 0.6526, 0.8032],
        [0.2479, 0.1951, 0.9805, 0.5539, 0.3997],
        [0.4069, 0.5350, 0.9098, 0.9688, 0.4189],
        [0.5871, 0.9368, 0.8115, 0.8433, 0.6553],
        [0.2272, 0.8550, 0.7863, 0.1270, 0.1430]]
    predict("model.pt", test_vec)