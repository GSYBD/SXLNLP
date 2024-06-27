import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

"""
实现一个自行构造的找规律(机器学习)任务
规律：vector是一个自定义的n维向量，判断其中每位的奇偶性
"""

"""
疑问？ ：想问下老师，为什么我算出的结果最后的预测正确率一直只能在50%左右;

"""


class torchDemo(nn.Module):
    def __init__(self, n_input):
        super(torchDemo, self).__init__()
        self.fc1 = nn.Linear(1, n_input + 1)
        self.fc2 = nn.Linear(n_input + 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# 随机生成一个n维向量，并判断奇偶性，并将其每项相加
def vector(n):
    vec = np.random.randint(low=1, high=1000, size=n)
    parity = []
    for element in vec:
        if element % 2 == 0:
            parity.append(1)
        else:
            parity.append(0)
    return vec, parity


# 批量生成样本
def batch_generate(batch_size, n):
    vec_list = []
    parity_list = []
    for i in range(batch_size):
        vec, parity = vector(n)
        vec_list.append(vec)
        parity_list.append(parity)
    return torch.FloatTensor(np.array(vec_list)), torch.FloatTensor(np.array(parity_list))


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, n_input):
    # 将模型设置为评估模式
    model.eval()
    batch_size = 100
    correct, error = 0, 0
    x, y = batch_generate(batch_size, n_input)
    with torch.no_grad():
        forecast_y = model(x.view(-1, 1))
        for y_f, y_t in zip(forecast_y, y.view(-1, 1)):
            if torch.sigmoid(y_f) >= 0.5 and int(y_t) == 1:
                correct += 1
            elif torch.sigmoid(y_f) < 0.5 and int(y_t) == 0:
                correct += 1
            else:
                error += 1
        print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + error)))
        return correct / (correct + error)


def main():
    number_rounds = 10  # 轮数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    n_input = int(input("请输入向量维度："))  # 输入向量维度

    # 建立模型
    model = torchDemo(n_input)
    # 选择优化函数  添加L2正则化weight_decay，防止过拟合
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 选择二分类交叉熵损失函数
    criterion = nn.BCEWithLogitsLoss()
    # 创建训练集
    train_x, train_y = batch_generate(train_sample, n_input)
    # 将数据展开为一维，以便用于训练
    train_x_one = train_x.view(-1, 1)
    train_y_one = train_y.view(-1, 1)
    # 记录模型结构和每轮损失值
    records = []
    # 创建数据集和数据加载器
    dataset = TensorDataset(train_x_one, train_y_one)
    # 创建数据加载器，指定每个批次的大小为n_input，并在每个epoch开始时打乱数据。
    dataloader = DataLoader(dataset, batch_size=n_input, shuffle=True)
    for i in range(number_rounds):
        # 设置模型为训练模式
        model.train()
        # 记录本轮损失值和
        watch_loss = []
        for x, y in dataloader:
            # 权重归零
            optimizer.zero_grad()
            # 得到预测值
            y_p = model(x)
            # 计算loss
            loss = criterion(y_p, y)
            # 计算梯度
            loss.backward()
            # 更新权重
            optimizer.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (i + 1, np.mean(watch_loss)))
        acc = evaluate(model, n_input)  # 测试本轮模型结果
        records.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "weight.pt")
    # 画图
    print(records)
    plt.plot(range(len(records)), [l[0] for l in records], label="acc")  # 画acc曲线
    plt.plot(range(len(records)), [l[1] for l in records], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    main()
