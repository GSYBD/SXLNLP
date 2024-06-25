"""
week2 : 实现简单的模型任务
找出1*5向量中最大的值
"""
import numpy as np
import torch
from torch import nn


class TorchModel(nn.Module):
    def __init__(self, hidden_size):
        super(TorchModel, self).__init__()
        # 线性层
        self.layer = nn.Linear(hidden_size, 5)
        # 激活函数
        # self.aviation = nn.Softmax()
        # 损失函数
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        # 输出: [batch_size, 5]
        x = self.layer(x)
        # 输出: [batch_size, 5]
        # y_pred = self.aviation(x)
        y_pred = x
        # 计算损失
        if y is not None:
            loss = self.loss(y_pred, y.long().view(-1))
            return loss
        else:
            return y_pred


def build_data(batch_simple):
    X = []
    Y = []
    for i in range(batch_simple):
        x = np.random.random(5)
        y = np.argmax(x)
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)


def eavaluate(model):
    x,y = build_data(100)
    y_pred = model(x)
    correct = 0
    # 调用模型
    with torch.no_grad():
        # 测试
        model.eval()
        for y_pred,y in zip(y_pred,y):
            if torch.argmax(y_pred) == y:
                correct += 1
        # 输出正确的个数和正确率
        print(f"correct的个数: {correct}")
        print(f"正确率: {correct / 100}")





def main():
    batch_size = 20
    batch_simple = 5000  # 样本数量
    hidden_size = 5
    epoch_size = 30  # 迭代次数
    # 数据
    x, y = build_data(batch_simple)
    dataset = torch.utils.data.TensorDataset(x, y)
    dataset_batch = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 模型
    model = TorchModel(hidden_size)
    lr = 0.001
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 训练
    for epoch in range(epoch_size):
        model.train()
        batch_loss = []
        for batch_iter in dataset_batch:
            batch_x, batch_y = batch_iter
            optimizer.zero_grad() # 梯度归0
            loss = model(batch_x, batch_y) # 计算loss
            loss.backward()# 反向传播
            optimizer.step()# 更新参数

            batch_loss.append(loss.item())
        eavaluate(model)  # 评估函数
        print(f"epoch: {epoch}, loss: {sum(batch_loss) / len(batch_loss)}")

    torch.save(model.state_dict(), "week2_work.pth")
    return




if __name__ == '__main__':
    main()
    # 加载模型
    # model = TorchModel(5)
    # model.load_state_dict(torch.load('week2_work.pth'))
    #
    # # 测试
    # model.eval()
    # x = torch.randn(1, 5)
    # y = model(x)
    # print(x)
    # print(torch.argmax(y))
