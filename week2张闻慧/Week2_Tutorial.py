# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
 
"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个10维向量，所有值在0到1之间
如果前五维的平均值大于后五维的平均值，并且至少有三个值大于0.5，则为类别1
如果前五维的平均值小于后五维的平均值，并且至少有三个值小于0.5，则为类别2
其他情况为类别0
"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 20)  # 线性层
        self.linear2 = nn.Linear(20, 3)  # 线性层

    def forward(self, x):
        x = torch.relu(self.linear1(x))  # (batch_size, input_size) -> (batch_size, out_feature)
        x = self.linear2(x)  # (batch_size, 8) -> (batch_size, 3)
        return x

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
def build_sample():
    x = np.random.random(10)
    if np.mean(x[:5]) > np.mean(x[5:]) and np.sum(x[:5] > 0.5) >= 3:
        return x, 1
    elif np.mean(x[:5]) < np.mean(x[5:]) and np.sum(x[5:] < 0.5) >= 3:
        return x, 2
    else:
        return x, 0

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
def evaluate(model, criterion):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    y_np = y.numpy()
    count_class1 = np.sum(y_np == 1)
    count_class2 = np.sum(y_np == 2)
    count_class0 = np.sum(y_np == 0)

    print(f"本次预测集中共有{count_class1}个类别1，{count_class2}个类别2，{count_class0}个类别0")

    correct, total = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        loss = criterion(y_pred, y)  # 计算损失
        _, predicted = torch.max(y_pred, 1)  # 获取预测类别
        correct += (predicted == y).sum().item()  # 正确预测的数量
        total += y.size(0)  # 总样本数量
    print("准确率：%f" % (correct / total))
    return correct / total

# 主函数，训练模型并评估
def main():
    # 配置参数
    epoch_num = 30  # 训练轮数
    batch_size = 30  # 每次训练样本个数
    train_sample = 6000  # 每轮训练总共训练的样本总数
    input_size = 10  # 输入向量维度
    learning_rate = 0.001  # 学习率
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
        acc = evaluate(model, criterion)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
        # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 10
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
    main()
    test_vec, test_gt = build_test_dataset(50)
    predict("model.pth", test_vec)
