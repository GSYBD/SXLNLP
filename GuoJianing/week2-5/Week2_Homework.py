# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt




class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size1) # 线性层
        self.loss = nn.functional.cross_entropy
    def forward(self, x, y=None):
        y_pred = self.linear(x)  
            return self.loss(y_pred, y)  
        else:
            return y_pred  # 输出预测结果



def build_sample():
    x = np.random.random(5)

    if x[0] == max(x):
        return x, 0
    elif x[1] == max(x):
        return x, 1
    elif x[2] == max(x):
        return x, 2
    elif x[3] == max(x):
        return x, 3
    elif x[4] == max(x):
        return x, 4



def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  
        for y_p, y_t in zip(y_pred, y):  
            if torch.argmax(y_p) == int(y_t):
                correct += 1 
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  
    batch_size = 20  
    train_sample = 5000  
    input_size = 5  
    hidden_size1 = 5
    learning_rate = 0.001  
    # 建立模型
    model = TorchModel(input_size, hidden_size1)
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
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.txt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


def predict(model_path, input_vec):
    input_size = 5
    hidden_size1 = 5
    model = TorchModel(input_size,hidden_size1)
    model.load_state_dict(torch.load(model_path))  
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec)) 
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res)) 


if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.txt", test_vec)
# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt




class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size1) # 线性层
        self.loss = nn.functional.cross_entropy
    def forward(self, x, y=None):
        y_pred = self.linear(x)  
            return self.loss(y_pred, y)  
        else:
            return y_pred  # 输出预测结果



def build_sample():
    x = np.random.random(5)

    if x[0] == max(x):
        return x, 0
    elif x[1] == max(x):
        return x, 1
    elif x[2] == max(x):
        return x, 2
    elif x[3] == max(x):
        return x, 3
    elif x[4] == max(x):
        return x, 4



def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  
        for y_p, y_t in zip(y_pred, y):  
            if torch.argmax(y_p) == int(y_t):
                correct += 1 
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  
    batch_size = 20  
    train_sample = 5000  
    input_size = 5  
    hidden_size1 = 5
    learning_rate = 0.001  
    # 建立模型
    model = TorchModel(input_size, hidden_size1)
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
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.txt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


def predict(model_path, input_vec):
    input_size = 5
    hidden_size1 = 5
    model = TorchModel(input_size,hidden_size1)
    model.load_state_dict(torch.load(model_path))  
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec)) 
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res)) 


if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.txt", test_vec)
