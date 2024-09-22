"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个3维向量，这三个数能否构成一个三角形，如果可以则为正样本，反之为负样本（结果：准确率不是很高等后续模型修改）

"""
# coding:utf8
import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt


#生成随机样本，样本的生成方法，代表了我们要学习的规律
#随机生成一个3维向量
def build_sample():
    # x = np.random.randint(0,100,3)
    x = np.random.random(3)
    # if x[0] > x[2]:
    #     return x, 1
    # else:
    #     return x, 0
    # print(x)
    if x[0] + x[1] > x[2] and x[0] + x[2] > x[1] and x[1] + x[2] > x[0]:
        return x,1
    else:
        return x,0

#随机生成一批样本
#正负样本均匀生成
def build_dataset(total_sample_num):
    X=[]
    Y=[]
    for i in range(total_sample_num):
        x,y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X),torch.FloatTensor(Y)

# X,Y = build_dataset(1000)
# print(X,Y)

#构建模型
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size,1)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss
    
    def forward(self, x,y = None):#有y是在模型训练，无y在预测
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None: return self.loss(y_pred,y)
        else: return y_pred

def main():
    #配置参数
    epoch_num = 20 #训练轮数
    batch_size = 20 # 每次训练样本个数
    train_sample = 5000 #每轮训练样本个数
    input_size = 3 #输入向量维度
    learning_rate = 0.001 #学习率
    #建立模型
    model = TorchModel(input_size)
    #选择优化器(权重更新)
    optim= torch.optim.Adam(model.parameters(),lr = learning_rate)
    log = []
    #创建训练集，正常任务是读取训练集
    train_x,train_y = build_dataset(train_sample)
    #训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample//batch_size):
            x = train_x[batch_index:(batch_index+1)*batch_size]
            y = train_y[batch_index:(batch_index+1)*batch_size]
            # print(x,y)
            loss = model(x,y) #计算loss
            loss.backward() #计算梯度
            optim.step() #更新权重
            optim.zero_grad() #清空梯度
            watch_loss.append(loss.item())
        print("--------\n第%d轮训练结束，平均loss为%f"%(epoch+1,np.mean(watch_loss)))
        acc = evaluate(model) #测试本轮模型结果
        log.append([acc,float(np.mean(watch_loss))])
    #保存模型
    torch.save(model.state_dict(),"model.pt")
    #画图
    print(log)
    plt.plot(range(len(log)),[i[0] for i in log],label="acc") #画acc曲线
    plt.plot(range(len(log)),[i[1] for i in log],label="loss") #画loss曲线
    plt.legend()
    plt.show()
    return

#测试代码
#用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x,y = build_dataset(test_sample_num)
    print("本次预测中有%d个正样本，%d个负样本"%(sum(y),test_sample_num-sum(y)))
    correct,wrong = 0,0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("本次预测中%d个样本预测正确，正确率：%f"%(correct,correct/(correct+wrong)))
    return correct/(correct + wrong)

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 3
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    # main()
    test_vec = [[0.07889086,0.15229675,0.31082123],
                [0.94963533,0.5524256,0.95758807],
                [0.78797868,0.67482528,0.13625847],
                [0.79349776,0.59416669,0.92579291]]
    predict("model.pt", test_vec)