import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()

        #线性层 设置为返回5个数
        self.linear = nn.Linear(input_size, 5) 

        #激活函数 sigmoid 在这里是为了将输出值归一化到0-1之间 使用交叉熵我们不需要激活函数
        #self.activation = torch.sigmoid #激活函数 sigmoid

        #损失函数 均方差 我用会使用交叉熵替代均方差
        #self.loss = nn.functional.mse_loss 

        #交叉熵损失函数
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        #让x经过线性层
        y_pred = self.linear(x)
        #y_pred = self.activation(x)
        if y is not None:
            #如果有真实标签 计算损失
            return self.loss(y_pred, y)
        else:
            #如果没有真实标签 返回预测值
            return y_pred

#生成一个样本
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)
    #print(x, sample_number)
    return x, max_index

#print(build_sample())


#生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)


#print(build_dataset(5))


def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    
    # 由于y已经是一维张量，包含最大值的索引，因此不需要对y进行one-hot编码处理
    # 直接统计各类样本的数量
    unique, counts = torch.unique(y, return_counts=True)
    counts_dict = dict(zip(unique.tolist(), counts.tolist()))
    print("本次预测集中有%d个样本," % test_sample_num +
          "".join([f"{i+1}型样本: {counts_dict.get(i, 0)}个, " for i in range(5)])[:-2])
    
    correct_predictions = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        y_pred = torch.argmax(y_pred, dim=1)  # 获取预测结果中最大值的索引
        correct_predictions = (y_pred == y).sum().item()  # 计算正确预测的数量
    
    accuracy = correct_predictions / test_sample_num
    print(f"正确预测个数: {correct_predictions}, 正确率: {accuracy:.6f}")
    return accuracy

# print(evaluate(TorchModel(5)))


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20 # 每次训练样本个数
    train_sample = 5000 # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01   # 学习率
    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器 权重的更新
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):    
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = y.long()  # 将y转换为整数类型
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
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
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.pt", test_vec)
