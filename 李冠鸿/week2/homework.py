import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random

"""

基于python框架编写模型训练
实现一个自行构造的找规律（机器学习）人物
规律：x是一个5维向量，第几个数最大就是第几类

"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.loss = nn.functional.cross_entropy


    def forward(self, x, y=None):
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y)
        else:
            return x

def build_sample():
    x = np.random.random(5)
    # 获取最大值的索引
    max_index = np.argmax(x)
    if max_index == 0:
        return x, 0
    elif max_index == 1:
        return x, 1
    elif max_index == 2:
        return x, 2
    elif max_index == 3:
        return x, 3
    else:
        return x, 4

# 生成一个5维向量，
# def build_sample():
#     x = np.random.random(5)
#     # list = [0,0,0,0,0]
#     # list[np.argmax(x)] = np.argmax(x) + 1
#     return x, torch.LongTensor([np.argmax(x)])

# 随机生成一批数据
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 第几类样本分别有多少个
def sum_category(list):
    dict = {"1" : 0, "2" : 0, "3" : 0, "4" : 0, "5" : 0}
    for i in list:
        if i+1 == 1:
            dict["1"] += 1
        elif i+1 == 2:
            dict["2"] += 1
        elif i+1 == 3:
            dict["3"] += 1
        elif i+1 == 4:
            dict["4"] += 1
        else:
            dict["5"] += 1
    return dict


# 测试代码
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    category = sum_category(y)
    print("本次预测集中共有%d个一类样本，%d个二类样本，%d个三类样本，%d个四类样本，%d个五类样本" % (category["1"], category["2"], category["3"], category["4"], category["5"]))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)   # 预测模型
        # print(y_pred)
        # print("=============")
        # print(y)
        for y_p, y_t in zip(y_pred, y):
            if y_p.argmax() == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d， 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20 # 每次训练样本个数
    train_sample = 5000 # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001

    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
    log = []
    # 创建训练集， 正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_size + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_size + 1) * batch_size]
            # y.flatten()
            # print("Shape of Y222:", y.shape)
            loss = model(x, y)  #计算loss
            loss.backward() # 计算梯度
            optim.step()    # 更新权重
            optim.zero_grad()   #梯度归零
            # print(loss)
            watch_loss.append(loss.item())
        print("=====\n第%d轮平均loss：%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model1.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label = "acc")
    plt.plot(range(len(log)), [l[1] for l in log], label = "loss")
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    # print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))

    for vec, res in zip(input_vec, result):
        # print("输入：%s，预测类别：%d，概率值：%d" % (vec, round(float(res)),res))
        print(torch.max(res))
        max_res = torch.max(res).item()
        print("输入：{}，预测类别：{}，概率值：{}".format(vec, round(float(max_res)), max_res))

if __name__ == '__main__':
    main()
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
    #             [0.94963533,0.5524256,0.95058807,0.95520434,0.84890681],
    #             [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    #
    # predict("model1.pt", test_vec)
























