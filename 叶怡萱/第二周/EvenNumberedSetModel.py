# coding:utf8

import torch#PyTorch深度学习库
import torch.nn as nn#神经网络相关模块
import numpy as np#科学计算库
import random
import json#用于处理JSON数据
import matplotlib.pyplot as plt#用于绘制图形

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量,如果第一个值和第二个值都是偶数,认为是偶数集;反之为非偶数集.
"""


class EvenNumberedSetModel(nn.Module):
    def __init__(self, input_size):
        super(EvenNumberedSetModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # 线性层nn.Linear
        self.activation = torch.sigmoid  # sigmoid归一化函数
        self.loss = nn.functional.mse_loss  # loss损失函数采用均方差损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:#如果给定了真实标签y
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个2维向量，如果第一个值和第二个值都是偶数,认为是偶数集;反之为非偶数集.
def build_sample():
    x = np.random.random(2)
    if x[0] / 2 == 0 and x[1] / 2 == 0:
        return x, 1
    else:
        return x, 0


# 随机生成一批样本
# 偶数集和非偶数集均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)
#将 X 和 Y 列表转换成 PyTorch 的 FloatTensor 格式,并将它们返回。这是因为 PyTorch 的神经网络模型通常需要 PyTorch 张量作为输入。


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100#设置了评估时使用的测试样本数量为100个。
    x, y = build_dataset(test_sample_num)#调用了 build_dataset() 函数,该函数会生成100个测试样本的输入 x 和输出标签 y
    print("本次预测集中共有%d个偶数集，%d个非偶数集" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0#初始化了两个计数器,用于记录正确预测和错误预测的数量。
    with torch.no_grad():#上下文管理器,用于在评估过程中关闭梯度计算,以提高计算效率。
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比,循环遍历了模型的预测输出 y_pred 和真实标签 y。
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 非偶数集判断正确, correct 计数器加1
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  # 偶数集判断正确
            else:
                wrong += 1 #模型的预测与真实标签不一致,则将 wrong 计数器加1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 2  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = EvenNumberedSetModel(input_size)#创建了一个名为 TorchModel 的模型,并将输入向量的维度设为5。
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)#选择了 Adam 优化器,并将学习率设为0.001
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)#调用 build_dataset() 函数,创建了5000个训练样本的输入 train_x 和输出标签 train_y。
    # 训练过程
    for epoch in range(epoch_num):
        model.train()#将模型切换到训练模式
        watch_loss = []
        for batch_index in range(train_sample // batch_size):  #这个循环将训练集划分为多个小批量,每个批次包含20个样本。  
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]#从训练集中取出当前批次的输入样本
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]#从训练集中取出当前批次的标签
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())#将每个批次的损失值记录下来
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果.调用 evaluate() 函数,评估当前训练轮次的模型在测试集上的准确率。
        log.append([acc, float(np.mean(watch_loss))])#将当前轮的准确率和平均损失值记录下来,用于后续绘图。

    # 保存模型
    torch.save(model.state_dict(), "model.pt")#将训练好的模型参数保存到文件 "model.pt" 中。
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线 绘制准确率随训练轮数的变化曲线。
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线 绘制损失值随训练轮数的变化曲线。
    plt.legend()#添加图例
    plt.show()#显示绘制的图像
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):# 训练好的模型权重文件的路径,需要预测的输入向量列表
    input_size = 5
    model = EvenNumberedSetModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 模型设置为评估(测试)模式,这样会关闭一些诸如 Dropout 之类的层,使得预测结果更加确定
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
#当我们使用 zip() 函数同时遍历两个列表时,它会自动将对应位置的元素配对,然后我们可以在 for 循环中分别访问这两个配对的元素。
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":#Python 的标准运行入口,确保 predict() 函数只在直接运行本脚本时被执行。
    main()
    #定义了一个测试向量 test_vec,并调用 predict() 函数进行预测
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.pt", test_vec)
