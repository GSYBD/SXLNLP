# coding:utf8

from typing import List
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律： 给定一个10维向量 x：
    如果 x 的第1个元素大于第10个元素，并且第5个元素与第6个元素的差的绝对值小于0.5，则 y = 1。
    在其他所有情况下，y = 0。

"""

class FindPatternsModel(nn.Module):
    def __init__(self, input_size):
        super(FindPatternsModel,self).__init__()
        self.linear1 = nn.Linear(input_size, 10) # 1*10  10*10 -> 1*10
        self.linear2 = nn.Linear(10,1) # 1*10  10*1  ->1*1
        self.activation = torch.sigmoid # 归一化函数
        #self.loss = nn.BCELoss() # 二进制交叉熵损失函数
        self.loss = nn.MSELoss() # 均方差损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self,x,y=None):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred

# 生成一个样本
def build_sample():
    # 给定一个10维向量 x：
    x = np.random.random(10)
    # 如果 x 的第1个元素大于第10个元素，并且第5个元素与第6个元素的差的绝对值小于0.5，则 y = 1。
    if x[0] > x[9] and(abs(x[4] - x[5]) < 0.5) :
        return x, [1]
    else:
        return x,[0]

# 随机生成一批样本
def build_dataset(totol_num):
    X = []
    Y = []
    for i in range(totol_num):
        x,y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.FloatTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval() # 测试模式
    test_num = 200
    x,y = build_dataset(test_num)
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_num - sum(y)))
    correct,wrong = 0, 0
    with torch.no_grad():# 不计算梯度
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y) : # 与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1 # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1 # 正样本判断正确
            else:
                wrong += 1
    accuracy = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy

def main():
    # 配置参数
    epoch_num = 100 # 训练轮数
    batch_size = 10 # 每次训练样本个数
    train_num = 5000 # 训练样本总数
    input_size = 10 # 输入向量维度
    learning_rate = 0.001 #学习率
    # 建立模型
    model = FindPatternsModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log = []
    # 创建训练集
    train_x, train_y = build_dataset(train_num)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_num // batch_size):
            train_b_begin = batch_index * batch_size
            train_b_end = (batch_index +1) * batch_size
            x = train_x[train_b_begin : train_b_end]
            y = train_y[train_b_begin :train_b_end]
            optim.zero_grad() # 梯度归0
            loss = model(x,y) # 计算loss
            loss.backward() # 计算梯度
            optim.step() # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model) # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "find_patterns_model.pth")
    # 画图
    print(log)
    plt.plot(range(len(log)),[l[0] for l in log], label="acc") # 画acc
    plt.plot(range(len(log)),[l[1] for l in log], label="loss") # 画loss
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 10
    model = FindPatternsModel(input_size)
    model.load_state_dict(torch.load(model_path)) # 加载训练好的权重

    model.eval() # 测试模式
    with torch.no_grad(): # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec)) # 模型预测
    for vec ,res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果

if __name__ == "__main__":
    main()
    test_vec: List[List[float]] = [
        [0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.908920843, 0.1963533, 0.5524256, 0.95758807, 0.95520434,
         0.24890681],
        [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681, 0.94963533, 0.5524256, 0.95758807, 0.95520434,
         0.84890681],
        [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.99871392, 0.94963533, 0.5524256, 0.95758807, 0.95520434,
         0.54890681],
        [0.1349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894, 0.94963533, 0.5524256, 0.95758807, 0.95520434,
         0.84890681]]
    predict("find_patterns_model.pth", test_vec)