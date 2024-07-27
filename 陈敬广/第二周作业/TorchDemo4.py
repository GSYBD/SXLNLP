import torch
import torch.nn as nn
import numpy as np

# import matplotlib.pyplot as plt

'''
规律：x是一个5维向量，值最大的维度
'''


# 搭建神经网络,继承基类nn.Module
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        # 两层线性网络，输出5维向量
        self.fc1 = nn.Linear(input_size, 8)
        self.fc2 = nn.Linear(8, 5)
        # 激活函数
        self.activate = torch.tanh
        # 损失函数：交叉熵
        self.loss = nn.CrossEntropyLoss()

    # 定义流程
    def forward(self, x, y=None):
        x = self.fc1(x)
        x = self.activate(x)
        y_pred = self.fc2(x)
        if y is not None:
            return self.loss(y_pred, torch.argmax(y, dim=1))
        else:
            return y_pred


# 生成样本值
def build_sample():
    x = np.random.random(5)
    y = np.zeros(5)
    y[np.argmax(x)] = 1
    return x, y


def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)

    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试每轮执行完以后正确率
def evaluate(model):
    # 模型测试
    model.eval()
    # 测试样本数据
    test_x, test_y = build_dataset(100)
    correct, wrong = 0, 0
    # 不计算梯度
    with torch.no_grad():
        y_pred = model.forward(test_x)
        for y_p, y_t in zip(y_pred, test_y):
            if np.argmax(y_p) == np.argmax(y_t):
                correct += 1
            else:
                wrong += 1

    print('正确预测个数: %d，正确率：%f' % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 训练过程
def main():
    epoch_num = 20  # 训练20轮
    batch_size = 20  # 每次训练样本数
    train_sample = 5000  # 训练样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 创建模型
    model = TorchModel(input_size)
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 样本数据
    train_x, train_y = build_dataset(train_sample)
    log = []
    # 开始训练
    for epoch in range(epoch_num):
        # 训练模式
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            # 损失差
            loss = model.forward(x, y)
            # 计算梯度
            loss.backward()
            # 更新权重
            optim.step()
            # 梯度清零
            optim.zero_grad()
            watch_loss.append(loss.item())
        print('第%d轮平均误差: %f' % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        # log.append([acc, float(np.mean(watch_loss))])
    torch.save(model.state_dict(), 'model2.pt')
    # 画图
    # plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    # plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    # plt.legend()
    # plt.show()
    return


def predict(model_path):
    model = TorchModel(5)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_x, test_y = build_dataset(100)
    with torch.no_grad():
        y_pred = model.forward(test_x)

    for vec, y_p in zip(test_x, y_pred):
        print(f'输入：{vec}, 预测类别：{y_p}')


if __name__ == '__main__':
    # main()
    # test_vec = [[0.8991, 0.0228, 0.6137, 0.3143, 0.1948],
    #             [0.1698, 0.0155, 0.3078, 0.8429, 0.6165],
    #             [0.5499, 0.7533, 0.6151, 0.4862, 0.5782],
    #             [0.1807, 0.3760, 0.1454, 0.1869, 0.1977],
    #             [0.6239, 0.6515, 0.3019, 0.0673, 0.2211]]
    predict("model2.pt")
