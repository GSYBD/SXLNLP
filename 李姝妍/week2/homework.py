import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层
        self.activation = nn.Softmax(dim=1)  # softmax归一化函数
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失

    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        y_pred = self.activation(x)  # (batch_size, num_classes)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果

def build_sample(num_classes):
    x = np.random.random(5)
    y = random.randint(0, num_classes - 1)
    return x, y

def build_dataset(total_sample_num, num_classes):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample(num_classes)
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model, test_sample_num, num_classes):
    model.eval()
    x, y = build_dataset(test_sample_num, num_classes)
    print("本次预测集中共有%d个样本" % test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        _, predicted = torch.max(y_pred.data, 1)
        correct += (predicted == y).sum().item()
        wrong += (predicted != y).sum().item()
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / test_sample_num))
    return correct / test_sample_num

def main():
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 3  # 分类类别数
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample, num_classes)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, train_sample, num_classes)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    print("1111",model.state_dict())
    print("222",torch.FloatTensor)
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

def predict(model_path, input_vec, num_classes):
    input_size = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    probabilities = torch.softmax(result, dim=1)
    for vec, res, prob in zip(input_vec, torch.argmax(result, dim=1), probabilities):
        print("输入：%s, 预测类别：%d, 概率值：%s" % (vec, res.item(), prob))  # 打印结果

if __name__ == "__main__":
    # main()
    test_vec = [[0.8855, 0.7251, 0.4502, 0.4430, 0.7012],
        [0.2741, 0.0580, 0.0501, 0.2263, 0.8054],
        [0.7845, 0.8311, 0.1785, 0.5571, 0.0440],
        [0.6263, 0.6524, 0.9491, 0.7828, 0.2628],]
    predict("model.pt", test_vec, 3)
