import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


"""
基于交叉熵的多分类任务
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.activation = torch.sigmoid
        # self.loss = nn.CrossEntropyLoss()
        self.loss = nn.functional.mse_loss

    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


# 构建数据集方法
def build_sample():
    x = np.random.random(5)
    if x[0] > x[4]:
        return x, 1
    else:
        return x, 0


# 构建数据集
def build_dataset(total_sample_num):
    X = []
    Y = []

    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])  #为什么这里y加[]
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num = 20
    batch_size = 20
    learning_rate = 0.001
    input_size = 5
    train_sample = 5000
    model = TorchModel(input_size)

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss = model(x, y)  # 模型输入x,y 返回loss
            loss.backward() #自动计算梯度
            optim.step() #优化权重
            optim.zero_grad() #梯度归零
            watch_loss.append(loss.item())  # 这里什么意思？
            print("=========\n第%d轮,平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
            acc = evaluate(model)
            log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), "model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载
    print(model.state_dict())

    model.eval()  #注意这个跟上面的model.train()区别 模式选择
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果
if __name__ == "__main__":
    main()
    test_vec = [[0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.79349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
    predict("model.pt", test_vec)
