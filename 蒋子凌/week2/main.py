"""
蒋子凌第二周作业
使用sklearn生成数据集，用pytorch构建具有两个线性层的神经网络，使用relu作为激活函数，交叉熵为损失函数，SGD为优化器，学习率0.01，训练轮数100

PC配置:
CPU: i7-12700HQ
RAM: 16GB
GPU: RTX3060

结果：
第一次：
Epoch [10/100], Loss: 1.1259
Epoch [20/100], Loss: 1.0547
Epoch [30/100], Loss: 1.0052
Epoch [40/100], Loss: 0.9677
Epoch [50/100], Loss: 0.9377
Epoch [60/100], Loss: 0.9128
Epoch [70/100], Loss: 0.8914
Epoch [80/100], Loss: 0.8725
Epoch [90/100], Loss: 0.8557
Epoch [100/100], Loss: 0.8404
Accuracy: 0.5900

第二次：
Epoch [10/100], Loss: 1.1113
Epoch [20/100], Loss: 1.0366
Epoch [30/100], Loss: 0.9870
Epoch [40/100], Loss: 0.9507
Epoch [50/100], Loss: 0.9220
Epoch [60/100], Loss: 0.8982
Epoch [70/100], Loss: 0.8778
Epoch [80/100], Loss: 0.8598
Epoch [90/100], Loss: 0.8437
Epoch [100/100], Loss: 0.8291
Accuracy: 0.6150
"""

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.optim as optim  # 导入 PyTorch 的优化器模块
import numpy as np  # 导入 NumPy 库，用于数组操作
from sklearn.model_selection import train_test_split  # 导入 sklearn 的数据集分割函数
from sklearn.datasets import make_classification  # 导入 sklearn 的生成分类数据集函数
from sklearn.metrics import accuracy_score  # 导入 sklearn 的计算准确度函数

# 数据集准备
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_informative=15, random_state=42)
# 生成一个包含 1000 个样本、20 个特征、3 个类别的数据集，其中 15 个特征是有信息的

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 将数据集分割为训练集和测试集，比例为 80% 和 20%

X_train = torch.tensor(X_train, dtype=torch.float32)  # 将训练集特征转换为 PyTorch 张量
X_test = torch.tensor(X_test, dtype=torch.float32)  # 将测试集特征转换为 PyTorch 张量
y_train = torch.tensor(y_train, dtype=torch.long)  # 将训练集标签转换为 PyTorch 张量
y_test = torch.tensor(y_test, dtype=torch.long)  # 将测试集标签转换为 PyTorch 张量


# 模型定义
class SimpleNN(nn.Module):  # 定义一个简单的神经网络类，继承自 nn.Module
    def __init__(self, input_size, num_classes):  # 初始化函数，定义网络结构
        super(SimpleNN, self).__init__()  # 调用父类的初始化函数
        self.fc1 = nn.Linear(input_size, 50)  # 定义第一层全连接层，有 50 个神经元
        self.fc2 = nn.Linear(50, num_classes)  # 定义第二层全连接层，输出类别数

    def forward(self, x):  # 定义前向传播函数
        x = torch.relu(self.fc1(x))  # 通过第一层全连接层，并使用 ReLU 激活函数
        x = self.fc2(x)  # 通过第二层全连接层，得到输出
        return x


input_size = X_train.shape[1]  # 输入特征的数量
num_classes = len(np.unique(y))  # 类别数
model = SimpleNN(input_size, num_classes)  # 实例化神经网络模型

# 训练模型
criterion = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 定义优化器为随机梯度下降，并设置学习率

num_epochs = 100  # 训练 100 个周期
for epoch in range(num_epochs):  # 迭代训练
    outputs = model(X_train)  # 前向传播，计算输出
    loss = criterion(outputs, y_train)  # 计算损失

    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播，计算梯度
    optimizer.step()  # 更新模型参数

    if (epoch + 1) % 10 == 0:  # 每 10 个周期打印一次损失
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 在评估模式中不计算梯度
    outputs = model(X_test)  # 前向传播，计算测试集输出
    _, predicted = torch.max(outputs.data, 1)  # 获取预测的类别
    accuracy = accuracy_score(y_test, predicted)  # 计算准确度
    print(f'Accuracy: {accuracy:.4f}')  # 打印准确度
