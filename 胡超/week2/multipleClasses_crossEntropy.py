# -*- coding: utf-8 -*-
"""
author: Chris Hu
date: 2024/6/27
desc:
sample
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")


# 定义模型
class MultiClassModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultiClassModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        y_pred = self.fc3(self.fc2(self.fc1(x)))
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred


def build_sample():
    x = np.random.random(5)
    y = [0 for _ in range(5)]
    y[np.argmax(x)] = 1
    return x, y


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    matrix_x = []
    matrix_y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        matrix_x.append(x)
        matrix_y.append(y)
    return torch.FloatTensor(np.array(matrix_x)), torch.FloatTensor(np.array(matrix_y))


def evaluate(model):
    model.eval()
    x, y = build_dataset(200)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if y_p.argmax() == y_t.argmax():
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 生成一些示例数据
    num_samples = 2000
    num_features = 5  # 每个样本包含五个值
    num_classes = 5  # 五个类别
    batch_size = 50
    # 创建数据集和数据加载器
    dataset = TensorDataset(*(build_dataset(num_samples)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # 初始化模型 创建模型
    model = MultiClassModel(num_features, num_classes)
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 训练模型
    num_epochs = num_samples // batch_size
    log = []
    watch_loss = []
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            loss = model(inputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {np.mean(watch_loss)}")
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    torch.save(model.state_dict(), "../stuff/multipleClassesModel.pt")
    plt.plot([i for i in range(len(log))], [acl[0] for acl in log], label="acc")  # 画acc曲线
    plt.plot([i for i in range(len(log))], [acl[1] for acl in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    print("Training complete.")


def predict(model_path, input_vec):
    num_features = 5  # 每个样本包含五个值
    num_classes = 5  # 五个类别
    model = MultiClassModel(num_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    # print(model.state_dict())
    model.eval()
    with torch.no_grad():
        results = model(torch.FloatTensor(input_vec))
    for vec, result in zip(input_vec, results):
        c = np.argmax(result).item()
        print("输入：%s, 预测类别：%d" % (vec, c))  # 打印结果


if __name__ == '__main__':
    main()
    test_vec = [[100, 1, 2, 3, 0.1],
                [100, 110, 100, 130, 100],
                [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                [0.79349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
    predict("../stuff/multipleClassesModel.pt", test_vec)
