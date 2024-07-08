# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练作业
基于交叉墒的多分类任务
规律：x是一个任意的5维向量，寻找5个数中最大的数

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层
        # self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉墒

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x) # (batch_size, input_size) - > (batch_size, num_classes)
        if y is not None:
            # 确保y是long类型，因为cross_entropy_loss期望target是long类型
            y = y.long()
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第n个值(n=1~5)大于其他的值，则第n个值返回正样本
def build_sample():
    '''
    以下是有问题的代码
    x = np.random.random(5)
    if max(x) == x[0]:
        return x, np.ndarray([1, 0, 0, 0, 0])
    elif max(x) == x[1]:
        return x, np.ndarray([0, 1, 0, 0, 0])
    elif max(x) == x[2]:
        return x, np.ndarray([0, 0, 1, 0, 0])
    elif max(x) == x[3]:
        return x, np.ndarray([0, 0, 0, 1, 0])
    else:
        return x, np.ndarray([0, 0, 0, 0, 1])
    :return:
    '''
    x = np.random.random(5)
    max_index = np.argmax(x) # 获取最大值的索引
    # 创建一个one-hot编码的向量，但只使用索引作为标签
    y_one_hot = np.zeros(5)
    y_one_hot[max_index] = 1
    return x, torch.tensor([max_index], dtype=torch.long)  # 返回类别索引


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    # correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        _,predicted = torch.max(y_pred, 1)  # 得到最大概率的索引
        correct = (predicted == y).sum().item()
        accuracy = correct / y.size(0)
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy
        # for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            # print('y_p is: ', y_p)
            # print('y_t is: ', y_t)


            # if max_index_p == max_index_t:
            #     correct += 1  # 样本判断正确
            # else:
            #     wrong += 1 #样本判断错误
            # if float(y_p) < 0.5 and int(y_t) == 0:
            #     correct += 1  # 负样本判断正确
            # elif float(y_p) >= 0.5 and int(y_t) == 1:
            #     correct += 1  # 正样本判断正确
            # else:
            #     wrong += 1
    # print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    # return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    num_classes = input_size
    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
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
    model = TorchModel(input_size, input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    # for vec, res in zip(input_vec, result):
    #     print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果
        # 获取预测类别的索引和概率值
    predicted_classes = torch.argmax(result, dim=1)
    probabilities = torch.max(result, dim=1).values

    for vec, pred_class, prob in zip(input_vec, predicted_classes, probabilities):
        # 将张量的元素转换为Python标量
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, pred_class.item(), prob.item()))


if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # test_vec = np.random.rand(4, 5)
    predict("model.pt", test_vec)