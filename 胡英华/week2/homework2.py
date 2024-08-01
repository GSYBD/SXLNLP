""" 
基于 pytorch 框架编写模型训练
实现一个自行构造的找规律（机器学习）任务
规律：x是一个5维向量，第几个数最大，就属于第几类
五分类任务
"""


import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) # (batch_size, input_size) -> (batch_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size) # (batch_size, hidden_size) -> (batch_size, output_size)
        self.loss = nn.functional.cross_entropy      # 交叉熵损失函数
    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear1(x)          
        y_pred = self.linear2(x)
        if y is not None:
            return self.loss(y_pred, y)  # 根据预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


# 生成一个样本，样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，第几个数最大，就属于第几类，也就是返回其索引
def build_sample():
    x = np.random.random(5)
    maxindex  = np.argmax(x)
    return x, maxindex


# 随机生成一批样本
# 正负样本均匀生成
# 生成的样本的个数由参数total_sample_num决定，然后一个样本为一个元组，包含一个5维向量和一个标签
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
def evaluate(model):
    model.eval()  # 设置模型为测试模式
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    #print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():  # 测试模式下，不计算梯度，节省内存
        y_pred = model(x)  # 计算预测值
        for y_p, y_t in zip(y_pred, y):
            max_value, max_index = torch.max(y_p, 0)
            if max_index == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 50 # 训练轮数
    batch_size = 20 # 每次训练的样本个数
    train_sample = 5000 # 每轮训练总共训练的样本总数
#    input_size = 5 # 输入向量维度
    learning_rate = 0.001 # 学习率

    # 建立模型
    model = TorchModel(5, 4, 5)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):  # 训练20轮
        model.train()  # 设置模型为训练模式
        watch_loss = [] 
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model.forward(x, y) # 计算损失  等价于 model.forward(x, y) backward()里面同时计算了y_pred和loss
            loss.backward()  # 计算梯度
            optim.step()       # 更新权重
            optim.zero_grad()  # 清空梯度
            watch_loss.append(loss.detach().numpy())   # loss是个张量
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model) # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
        
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss") # 画loss曲线
    plt.legend()
    plt.show()
    return


# # 使用训练好的模型做预测
def predict(model_path, input_vec):
    model = TorchModel(5, 4, 5)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        max_value, class_num = torch.max(res, 0)
        print("输入：%s, 预测类别：%d" % (vec, class_num))  # 打印结果
        
    
if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.pt", test_vec)    
        





