# coding:utf8

from typing import List
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个基于交叉熵的多分类任务
规律： 给定一个10维向量 x：
    1,2元素为第一组，3,4为第二组，564为第三组，7,8为第四组，9,10为第五组
    每组内元素相加，哪组值大则为第几类，即5分类任务

"""

class ClassifyModel(nn.Module):
    def __init__(self, input_size):
        super(ClassifyModel,self).__init__()
        self.linear = nn.Linear(input_size, 10) # 1*10  10*10 -> 1*10
        self.classify = nn.Linear(10, 5) # 1*10  10*10  ->1*5 5分类
        self.loss = nn.CrossEntropyLoss() # 交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred

# 生成一个样本
def build_sample():
    # 给定一个10维向量 x：
    x = np.random.random(10)
    # 1,2元素为第一组，3,4为第二组，564为第三组，7,8为第四组，9,10为第五组 哪一组值相加大则为第几类
    y = np.zeros(5)
    group = [x[0] + x[1], x[2] + x[3], x[4] + x[5], x[6] + x[7], x[8] + x[9]]
    index = np.argmax(group)
    y[index] = 1
    return x, y 

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
    correct,wrong = 0, 0
    with torch.no_grad():# 不计算梯度
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y) : # 与真实标签进行对比
            classify_p = int(y_p.argmax())
            classify_t = int(y_t.argmax())
            if classify_p == classify_t:
                correct += 1
            else:
                wrong += 1
    accuracy = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy

def main():
    # 配置参数
    epoch_num = 20 # 训练轮数
    batch_size = 20 # 每次训练样本个数
    train_num = 5000 # 训练样本总数
    input_size = 10 # 输入向量维度
    learning_rate = 0.001 #学习率
    # 建立模型
    model = ClassifyModel(input_size)
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
    torch.save(model.state_dict(), "classify.pth")
    # 画图
    plt.plot(range(len(log)),[l[0] for l in log], label="acc") # 画acc
    plt.plot(range(len(log)),[l[1] for l in log], label="loss") # 画loss
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 10
    model = ClassifyModel(input_size)
    model.load_state_dict(torch.load(model_path)) # 加载训练好的权重

    model.eval() # 测试模式
    with torch.no_grad(): # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec)) # 模型预测
    for vec ,res in zip(input_vec, result):
        classify_predict = int(res.argmax())
        print("输入：%s, 预测为第:%d组" % (vec, classify_predict + 1))  # 打印结果

if __name__ == "__main__":
    main()
    test_vec: List[List[float]] = [
        [0.97889086, 0.15229675, 0.31082123, 0.03504317, 0.108920843, 0.1963533, 0.5524256, 0.45758807, 0.75520434,
         0.24890681],
        [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681, 0.94963533, 0.5524256, 0.59758807, 0.35520434,
         0.84890681],
        [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.99871392, 0.94963533, 0.5524256, 0.95758807, 0.95520434,
         0.54890681],
        [0.1349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894, 0.94963533, 0.9524256, 0.97758807, 0.95520434,
         0.84890681],
        [0.1249776, 0.59416669, 0.92579291, 0.41567412, 0.7358894, 0.24963533, 0.9524256, 0.57758807, 0.95520434,
         0.98890681],
        [0.9249776, 0.99416669, 0.92579291, 0.41567412, 0.7358894, 0.24963533, 0.9524256, 0.57758807, 0.95520434,
         0.48890681]]
    predict("classify.pth", test_vec)