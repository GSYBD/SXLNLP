# coding:utf-8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律（机器学习）任务
规律：x是一个5维向量，如果第1个与第2个数的和>=第4个与第5个数的和，
则为正样本1，反之为负样本

"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.activation = nn.Sigmoid()
        self.loss = nn.functional.mse_loss

    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred
        

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第1个与第2个数的和>=第4个与第5个数的和，则为正样本1，反之为负样本
def build_sample():
    x = np.random.random(5)
    if x[0] + x[1] >= x[-1] + x[-2]:
        return x, 1
    else:
        return x, 0
    

# 随机生成一批样本
# 正负样本均匀生成
def build_samples(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 1000
    x, y = build_samples(test_sample_num)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num- sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if float(y_p) < 0.5 and y_t == 0:
                correct += 1
            elif float(y_p) >= 0.5 and y_t == 1:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20 # 训练轮数
    batch_size = 20 # 每次训练的样本数
    train_sample_num = 5000 # 训练样本数
    input_size = 5 # 输入层维度
    learning_rate = 0.001 # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 定义优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_samples(train_sample_num)
    # 训练模型
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample_num // batch_size):    
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
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
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()xzxxzZZXZZZZ
    test_vec = [[0.9319, 0.4134, 0.3695, 0.3831, 0.0413],
        [0.7872, 0.0491, 0.5005, 0.0846, 0.1573],
        [0.3534, 0.3122, 0.1802, 0.7283, 0.7162],
        [0.1936, 0.1171, 0.5924, 0.6862, 0.1791],
        [0.5275, 0.2337, 0.9954, 0.5569, 0.3829]]
    predict("model.pt", test_vec)
