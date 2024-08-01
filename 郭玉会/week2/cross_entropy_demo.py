import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练代码
实现一个分类任务
规律：x是一个5维向量，如果第一个数最大，则为第0类，第二个数最大，则为第1类，以此类推，第5个数最大，则为第4类
"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)   # 线性层，输入一个5维向量，输入一个标记分类的5维向量
        self.loss = nn.functional.cross_entropy # loss函数采用交叉熵

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred
        
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)
    y = max_index

    return x, y


def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct = 0
    wrong = 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.001

    # 建立模型
    model = TorchModel(input_size)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]

            loss = model(x, y)  # 计算loss
            loss.backward()   # 计算梯度
            optim.step()    # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("===========\n第%d轮 平均loss: %f" % (epoch + 1, np.mean(watch_loss)))

        acc = evaluate(model)  # 测试本轮模式结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "cross_entropy_model.pt")
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
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))

    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%s, 预测值：%s" % (vec, torch.argmax(res), res))  # 打印结果

if __name__ == "__main__":
    main()

    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]

    # predict("cross_entropy_model.pt", test_vec)