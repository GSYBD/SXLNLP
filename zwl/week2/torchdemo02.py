# -*- coding: utf-8 -*-
# name torchdemo02.py
# date 2024/6/29 23:40
"""

 10 个特征值（即 x 中的每一行）来准确预测其所属的类别（0、1 或 2，即 y 中的值)

"""
import torch
import torch.nn as nn

# 假设我们有一批数据，输入特征 x 和对应的类别标签 y
x = torch.randn(100, 10)  # 生成一个形状为(100, 10)的随机张量x，模拟100个样本，每个样本有10个特征
y = torch.randint(0, 3, (100,))  # 生成一个形状为(100,)的随机张量y，模拟100个样本的类别标签，取值为0、1、2

# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(10, 50)  # 定义第一层线性层，将输入的10个特征映射到50个维度
        self.layer2 = nn.Linear(50, 3)  # 定义第二层线性层，将50个维度映射到3个类别

    def forward(self, x):
        x = torch.relu(self.layer1(x))  # 对第一层的输出应用ReLU激活函数，增加非线性
        x = self.layer2(x)  # 第二层的输出，得到最终的预测类别
        return x

# 创建模型实例
model = NeuralNetwork()

# 定义损失函数（交叉熵）和优化器
loss_func = nn.CrossEntropyLoss()  # 选择交叉熵损失函数，适用于多分类问题
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 使用随机梯度下降优化器，优化模型的参数，学习率为0.01

#训练模型
for epoch in range(100):  # 训练100个轮次
    outputs = model(x)  # 模型对输入x进行前向传播，得到预测输出
    loss = loss_func(outputs, y)  # 计算预测输出和真实标签y之间的交叉熵损失
    optimizer.zero_grad()  # 清空之前的梯度
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 根据梯度更新模型的参数
    optimizer.step()

# 准备 5 组新的测试数据
new_x = torch.randn(5, 10)  # 5 个样本，每个样本 10 个特征

# 进行预测
with torch.no_grad():  # 关闭梯度计算，因为只是预测，不需要计算梯度
    predictions = model(new_x)

# 得到预测结果的类别
predicted_classes = torch.argmax(predictions, dim=1)

# 打印预测的类别
print("预测的类别:", predicted_classes)