import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


"""

分类规则：对于一个5维向量x, 它的label是这个向量中绝对值最小那个元素，即y = argmin(abs(x))
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 15)  # 线性层
        self.linear2 = nn.Linear(15, 5)  # 线性层
        self.activation = torch.relu  # sigmoid归一化函数
        self.loss = nn.functional.cross_entropy  # loss函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x1 = self.activation(self.linear1(x))
        y_pred = self.linear2(x1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


class Data(Dataset):
  def __init__(self, X_train, y_train):
    self.X = torch.from_numpy(X_train.astype(np.float32))
    self.y = torch.from_numpy(y_train).type(torch.LongTensor)
    self.len = self.X.shape[0]
  
  def __getitem__(self, index):
    return self.X[index], self.y[index]
  def __len__(self):
    return self.len


def build_dataset(n_samples ,n_features=20 ,n_classes=5):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test
    

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, testloader):
    model.eval()
    correct, wrong = 0, 0
    with torch.no_grad():
        for x_test, y_test in testloader:
            y_pred = model(x_test)  # 模型预测
            y_pred = torch.argmax(y_pred, dim=1)
            correct += (y_pred == y_test).sum().item()
            wrong += (y_pred != y_test).sum().item()
    print("测试集准确率：%f" % (correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 15  # 训练轮数
    batch_size = 500  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    input_size = 20  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    X_train, X_test, y_train, y_test = build_dataset(train_sample)
    traindata = Data(X_train, y_train)
    testdata = Data(X_test, y_test)
    trainloader = DataLoader(traindata, batch_size=batch_size, 
                         shuffle=True, num_workers=2)
    testloader = DataLoader(testdata, batch_size=batch_size)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        
        for x, y in trainloader:
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, testloader=testloader)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
    
