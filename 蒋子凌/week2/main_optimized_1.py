"""
蒋子凌第二周作业-优化
使用sklearn生成数据集，用pytorch构建具有三个线性层的神经网络，使用relu作为激活函数，交叉熵为损失函数，Adam为优化器，学习率0.001，训练轮数200
在main.py的基础上进行优化：增加了具有更多神经元的线性层，提高了训练轮数，调整了学习率，使用了不同的优化器，加入正则化防止过拟合。

PC配置:
CPU: i7-12700HQ
RAM: 16GB
GPU: RTX3060

结果：
第一次：
Epoch [10/200], Loss: 1.0043
Epoch [20/200], Loss: 0.9015
Epoch [30/200], Loss: 0.8084
Epoch [40/200], Loss: 0.7509
Epoch [50/200], Loss: 0.6827
Epoch [60/200], Loss: 0.6369
Epoch [70/200], Loss: 0.6069
Epoch [80/200], Loss: 0.5616
Epoch [90/200], Loss: 0.5231
Epoch [100/200], Loss: 0.4803
Epoch [110/200], Loss: 0.4772
Epoch [120/200], Loss: 0.4362
Epoch [130/200], Loss: 0.4077
Epoch [140/200], Loss: 0.3883
Epoch [150/200], Loss: 0.3812
Epoch [160/200], Loss: 0.3656
Epoch [170/200], Loss: 0.3306
Epoch [180/200], Loss: 0.3356
Epoch [190/200], Loss: 0.3267
Epoch [200/200], Loss: 0.2797
Accuracy: 0.7800

第二次：
Epoch [10/200], Loss: 0.9885
Epoch [20/200], Loss: 0.8682
Epoch [30/200], Loss: 0.7977
Epoch [40/200], Loss: 0.7217
Epoch [50/200], Loss: 0.6700
Epoch [60/200], Loss: 0.6145
Epoch [70/200], Loss: 0.5888
Epoch [80/200], Loss: 0.5340
Epoch [90/200], Loss: 0.5124
Epoch [100/200], Loss: 0.5030
Epoch [110/200], Loss: 0.4649
Epoch [120/200], Loss: 0.4480
Epoch [130/200], Loss: 0.4179
Epoch [140/200], Loss: 0.4039
Epoch [150/200], Loss: 0.3516
Epoch [160/200], Loss: 0.3561
Epoch [170/200], Loss: 0.3218
Epoch [180/200], Loss: 0.3345
Epoch [190/200], Loss: 0.3096
Epoch [200/200], Loss: 0.3198
Accuracy: 0.7550
"""
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.optim as optim  # 导入PyTorch的优化器模块
import numpy as np  # 导入NumPy库，用于数组操作
from sklearn.model_selection import train_test_split  # 导入数据集分割函数
from sklearn.datasets import make_classification  # 导入生成分类数据集的函数
from sklearn.metrics import accuracy_score  # 导入计算准确度的函数

# 数据集准备
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_informative=15, random_state=42)
# 生成一个有1000个样本、20个特征、3个类别的分类数据集

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 将数据集划分为训练集和测试集，比例为80%和20%

X_train = torch.tensor(X_train, dtype=torch.float32)  # 将训练集特征转换为PyTorch张量
X_test = torch.tensor(X_test, dtype=torch.float32)  # 将测试集特征转换为PyTorch张量
y_train = torch.tensor(y_train, dtype=torch.long)  # 将训练集标签转换为PyTorch张量
y_test = torch.tensor(y_test, dtype=torch.long)  # 将测试集标签转换为PyTorch张量


# 模型定义
class SimpleNN(nn.Module):  # 定义一个简单的神经网络类，继承自nn.Module
    def __init__(self, input_size, num_classes):  # 初始化函数，定义网络结构
        super(SimpleNN, self).__init__()  # 调用父类的初始化函数
        self.fc1 = nn.Linear(input_size, 100)  # 定义第一层全连接层，有100个神经元
        self.fc2 = nn.Linear(100, 50)  # 定义第二层全连接层，有50个神经元
        self.fc3 = nn.Linear(50, num_classes)  # 定义第三层全连接层，输出类别数
        self.dropout = nn.Dropout(0.5)  # 定义Dropout层，防止过拟合

    def forward(self, x):  # 定义前向传播函数
        x = torch.relu(self.fc1(x))  # 通过第一层全连接层，并使用ReLU激活函数
        x = self.dropout(x)  # 通过Dropout层
        x = torch.relu(self.fc2(x))  # 通过第二层全连接层，并使用ReLU激活函数
        x = self.fc3(x)  # 通过第三层全连接层，得到输出
        return x


input_size = X_train.shape[1]  # 输入特征的数量
num_classes = len(np.unique(y))  # 类别数
model = SimpleNN(input_size, num_classes)  # 实例化神经网络模型

# 训练模型
criterion = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 定义优化器为Adam，并设置学习率

num_epochs = 200  # 训练200个周期
for epoch in range(num_epochs):  # 迭代训练
    model.train()  # 设置模型为训练模式
    outputs = model(X_train)  # 前向传播，计算输出
    loss = criterion(outputs, y_train)  # 计算损失

    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播，计算梯度
    optimizer.step()  # 更新模型参数

    if (epoch + 1) % 10 == 0:  # 每10个周期打印一次损失
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 在评估模式中不计算梯度
    outputs = model(X_test)  # 前向传播，计算测试集输出
    _, predicted = torch.max(outputs.data, 1)  # 获取预测的类别
    accuracy = accuracy_score(y_test, predicted)  # 计算准确度
    print(f'Accuracy: {accuracy:.4f}')  # 打印准确度
