"""

鸢尾花分类
    基于sklearn.datasets模块提供的鸢尾花数据，进行分类预测
    由数据可知，分为0，1，2三类
    建立神经网络模型，进行训练

"""
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import sklearn.datasets as sd                               #数据加载
import matplotlib.pyplot as plt                             #画图模块
import sklearn.model_selection as ms                        #模型选择模块
import torch.optim as optim                                 #优化器
from torch.utils.data import DataLoader, TensorDataset      #数据初始化
from sklearn.preprocessing import StandardScaler
import time


def data_load():
    iris = sd.load_iris()               #加载鸢尾花数据

    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target

    x = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    x = x.astype(np.float32)
    y = y.astype(np.int64)
    train_x, test_x, train_y, test_y = ms.train_test_split(x, y,            #划分训练集和测试集
                                                           test_size=0.1,
                                                           random_state=10,
                                                           stratify=y)

    # #特征缩放，消除不同特征之间的尺度差异
    transfer = StandardScaler()
    train_x = transfer.fit_transform(train_x)
    test_x = transfer.transform(test_x)

    #将训练集和测试集分为两个dataset，同时转换成张量的形式
    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.tensor(train_y.values))
    test_dataset = TensorDataset(torch.from_numpy(test_x), torch.tensor(test_y.values))

    # print(data)                       #查看data的结构
    # return data                       #返回data，带入show_picture函数可以查看花瓣的分布

    #返回训练集、测试集、输入维度(x的行)、输出维度(y的个数)
    return train_dataset, test_dataset, train_x.shape[1], len(np.unique(y))


def show_picture(data):                 #提供可视化函数接口，可以调用来查看鸢尾花种类分布情况
    # 萼片可视化
    plt.figure('SEPAL')  # 分窗口显示
    plt.scatter(data['sepal length (cm)'],
                data['sepal width (cm)'],
                c=data['target'],
                cmap='brg')
    plt.colorbar()  # 颜色条

    # 花瓣可视化
    plt.figure('PETAL')
    plt.scatter(data['petal length (cm)'],
                data['petal width (cm)'],
                c=data['target'],
                cmap='brg')
    plt.colorbar()
    plt.show()              #根据图的规律，可以划分为0，1，2三分类


class IrisModule(nn.Module):                #网络模型
    def __init__(self, input_size, output_size):
        super(IrisModule, self).__init__()
        self.linear1 = nn.Linear(input_size, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, output_size)

    def _activation(self, x):               #激活函数sigmoid
        return torch.sigmoid(x)

    def forward(self, x):                   #前向传输

        x = self._activation(self.linear1(x))
        x = self._activation(self.linear2(x))
        x = self._activation(self.linear3(x))
        out_put = self.linear4(x)

        return out_put


def train():

    torch.manual_seed(10)       #选择随机种子，保证模型稳定

    module = IrisModule(input_size, output_size)

    loss = nn.CrossEntropyLoss()    #损失函数，交叉熵

    optimizer = optim.Adam(module.parameters(), lr=0.0001)        #Adam优化器

    epoch_num = 300                     #简单训练三百次

    for epoch in range(epoch_num):
        dataloader = DataLoader(train_data, shuffle=True, batch_size=9)     #初始化数据
        start = time.time()                                                 #可以查看每轮训练时间
        total_loss = 0.0            #总的损失
        total_num = 0               #参与计算损失的个数

        for x, y in dataloader:
            output = module(x)
            each_loss = loss(output, y)         #计算损失
            optimizer.zero_grad()               #梯度清零
            each_loss.backward()                #反向传播
            optimizer.step()                    #梯度更新

            total_num += len(y)
            total_loss += each_loss

        print('epoch: %4s loss: %.5f, time: %.2fs' %
              (epoch + 1, total_loss / total_num, time.time() - start))  # 第xx轮，平均损失值为xx，消耗时间xx

    torch.save(module.state_dict(), '../iris-predict-module.pt')        #保存模型


def predict():
    module = IrisModule(input_size, output_size)
    module.load_state_dict(torch.load('../iris-predict-module.pt'))

    dataloader = DataLoader(test_data, batch_size=5, shuffle=False)

    # 评估测试集,简单评估一下准确率
    correct = 0
    for x, y in dataloader:
        output = module(x)
        y_pred = torch.argmax(output, dim=1)
        correct += (y_pred == y).sum()

    print('准确率: %.5f' % (correct.item() / len(test_data)))


if __name__ == '__main__':
    train_data, test_data, input_size, output_size = data_load()
    train()
    predict()
