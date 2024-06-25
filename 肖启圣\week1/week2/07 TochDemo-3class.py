import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from collections import Counter


class Torchmodel(nn.Module):
    def __init__(self, input_size):
        super(Torchmodel, self).__init__()
        self.linear = nn.Linear(input_size, 3)
        self.activation = torch.sigmoid
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x)

        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def build_sample():
    x = np.random.random(3)
    if x[0] > x[1] and x[0] > x[2]:
        return x, 0
    elif x[1] > x[0] and x[1] > x[2]:
        return x, 1
    elif x[2] > x[0] and x[2] > x[1]:
        return x, 2


def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


def evaluate(model):
    model.eval
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    y_count = Counter(y.numpy())  # Counter计算每个数出现的次数，用字典存储

    print("本次预测集中第一个数最大的样本有%d，本次预测集中第二个数最大的样本有%d，本次预测集中第三个数最大的样本有%d" % (y_count[0], y_count[1], y_count[2]))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if float(y_p[0]) > 0.5 and float(y_p[1]) < 0.5 and float(y_p[2]) < 0.5 and int(y_t) == 0:
                correct += 1
            elif float(y_p[1]) > 0.5 and float(y_p[0]) < 0.5 and float(y_p[2]) < 0.5 and int(y_t) == 1:
                correct += 1
            elif float(y_p[2]) > 0.5 and float(y_p[0]) < 0.5 and float(y_p[1]) < 0.5 and int(y_t) == 2:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d,正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def predict(model_path, input_vec):
    input_size = 3
    model = Torchmodel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        b = np.argmax(res)  # 取出最大值索引
        print("输入为:%s,预测类别为：%d" % (vec, int(b)))


def main():
    epoch_num = 100
    batch_size = 20
    train_sample = 5000
    input_size = 3
    learning_rate = 0.001
    # 建立模型
    model = Torchmodel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss，我理解就是调用了forward函数
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("---------\n第%d轮平均loss：%f" % (epoch + 1, np.mean(watch_loss)))

        acc = evaluate(model)  # 测试本轮模型
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.path")
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #main()

    test_vec = [[0.34, 0.21, 0.54],
                [0.22, 0.96, 0.54],
                [0.56, 0.33, 0.21],
                [0.34, 0.76, 0.11]]
    predict("model.path", test_vec)
