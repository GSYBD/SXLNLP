#!/user/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim

# 数据集
batch_size = 1024
transform = transforms.Compose([
    transforms.ToTensor(),      # 图像转为张量
    transforms.Normalize((0.1307,), (0.3081,))  # 根据均值和标准差，进行归一化处理
])
# 训练集
train_dataset = datasets.MNIST(root=r'learn_torch/mnist/',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
# 测试集
test_dataset = datasets.MNIST(root=r'learn_torch/mnist/',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, 10)
        self.activate = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)  # 改变张量的形状，将（64，1，28，28）的数据变成（64，784）的数据，-1其实就是自动获取mini_batch，
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        x = self.activate(self.linear4(x))
        return self.linear5(x)


# 模型实例化
model = Net()
# 损失函数
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
# 优化器
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# 训练
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets = data
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 输出
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d,%5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


# 测试
def model_test():
    correct = 0  # 正确数据
    total = 0  # 全部数据
    with torch.no_grad():  # 不需要计算梯度
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy on test set:%d %%' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(40):
        train(epoch)
        model_test()
