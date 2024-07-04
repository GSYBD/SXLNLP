# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

'''基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个2维向量，第一元素+第二元素%2 = 0为第一类否则第二类
'''


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 20)  # 线性层
        self.linear2 = nn.Linear(20, 2)  # 改为2，因为有两个类别
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.Sigmoid(x)
        x = self.linear2(x)
        return x


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
def build_sample():
    # 生成一个2维随机向量
    x = np.random.randint(0, 11, 2)

    if x[0] + x[1] % 2 == 0:
        return x, 0
    else:
        return x, 1


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    print("本次预测集中共有%d个第一类，%d个第二类" % (sum(y), test_sample_num - sum(y)))

    correct, wrong = 0, 0

    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        _, predicted = torch.max(y_pred, 1)
        correct += (predicted == y).sum().item()  # 正确预测数量
        wrong += (predicted != y).sum().item()  # 错误预测数量
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 主函数，训练模型并评估
def main():
    # 配置参数
    epoch_num = 2000  # 训练轮数
    batch_size = 30  # 每次训练样本个数
    train_sample = 6000  # 每轮训练总共训练的样本总数
    input_size = 2  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    log = []
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            optimizer.zero_grad()  # 梯度归零
            y_pred = model(x)  # 前向传播
            loss = criterion(y_pred, y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            watch_loss.append(loss.item())
        print("第%d轮 平均loss: %f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
        # 保存模型
    torch.save(model.state_dict(), "mymodel.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 2
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度
        result = model(torch.FloatTensor(input_vec))  # 模型预测
        _, predicted = torch.max(result, 1)  # 获取预测类别
    for vec, res in zip(input_vec, predicted):
        print("输入：%s, 预测类别：%d" % (vec, res.item()))  # 打印结果


# 生成测试集并预测
def build_test_dataset(total_test_num):
    X = []
    Y = []
    for i in range(total_test_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return X, Y


if __name__ == "__main__":
    # main()
    test_vec, test_gt = build_test_dataset(50)
    predict("mymodel.pt", test_vec)
