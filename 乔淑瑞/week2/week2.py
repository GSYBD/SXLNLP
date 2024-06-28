import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 定义一个简单的多分类模型
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.softmax = nn.Softmax(dim=1)  # softmax激活函数
        self.loss = nn.functional.mse_loss  # 损失函数均方差

    def forward(self, x, y=None):
        x = self.linear(x)
        x = self.softmax(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


# 生成样本数据
def build_dataset(total_sample_num, input_size):
    X = []
    Y = []
    for i in range(total_sample_num):
        x = np.random.random(input_size)  # 生成一个随机向量
        max_index = np.argmax(x)  # 找到最大值的索引
        Y.append(max_index)  # 标签为最大值的索引
        X.append(x)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试模型准确率
def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        correct = (torch.argmax(y_pred, dim=1) == y).sum().item()
        accuracy = correct / len(y)
    print(f"正确预测个数：{correct}, 正确率：{accuracy}")
    return accuracy


# 主函数，训练模型并进行测试
def main():
    # 配置参数
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.001

    # 建立模型
    model = TorchModel(input_size)

    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 创建训练集
    train_x, train_y = build_dataset(train_sample, input_size)

    # 训练过程
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(0, train_sample, batch_size):
            x_batch = train_x[batch_index:batch_index + batch_size]
            y_batch = train_y[batch_index:batch_index + batch_size]

            # 前向传播及计算损失
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = nn.functional.cross_entropy(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            watch_loss.append(loss.item())

        print(f"第{epoch + 1}轮平均loss: {np.mean(watch_loss)}")

        # 每轮结束测试模型准确率
        acc = evaluate(model, train_x, train_y)
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.pt")

    # 画图
    log = np.array(log)
    plt.plot(range(len(log)), log[:, 0], label="acc")
    plt.plot(range(len(log)), log[:, 1], label="loss")
    plt.legend()
    plt.show()


# 执行主函数
if __name__ == "__main__":
    main()
