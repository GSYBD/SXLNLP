# coding:utf8

import torch
import torch.nn as nn;
import numpy as np;
import random
import json
import matplotlib.pyplot as plt
import torch.nn.functional

"""
基于pytorch框架实现基于交叉熵的多分类任务
任务描述：创建一个数据集，X是二维张量 [[a1,b1],[a2,b2]...[an,bn]],Y有3中分类情况，设a1+b1=A，当0<A<5时属于类别0,当5<=A<10时属于1，当10<=A<20时属于类别2
0 < a1 < 10,  0 < b1 < 10
"""


num_classes = 3

class MyCossEntropy(nn.Module):
    def __init__(self,input_size):
        super(MyCossEntropy, self).__init__()
        self.liner = nn.Linear(input_size,num_classes)
        #self.liner2 = nn.Linear(hidden_size,output_size)
        # 选择softmax为激活函数
        self.activation = torch.softmax
        # 选择交叉熵损失函数
        self.loss= torch.nn.functional.cross_entropy
        
    def forward(self, x, y=None):
        # 计算交叉熵损失
        x = self.liner(x)
        y_pred = self.activation(x,dim=1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果
    
    
# 随机生成一批样本
# 正负样本均匀生成
# def build_Sample():

#生成数据集
def build_dataSet(total_sample_num):
    X=[]
    Y=[]
    for i in range(total_sample_num):
        x1 = random.uniform(0,10)
        x2 = random.uniform(0,10)
        X.append([x1,x2])
        if x1+x2 < 5:
            Y.append(0)
        elif x1+x2 < 10:
            Y.append(1) 
        else:
            Y.append(2)
    
    # 转换为张量
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate( model ):
    # 创建数据集
    model.eval()
    test_sample_num = 100
    x,y = build_dataSet( test_sample_num )
    correct = 0
    wrong = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if ( int(y_t) == 0 and y_p[0] > y_p[1] and y_p[0] > y_p[2] ):
                correct += 1
            elif ( int(y_t) == 1 and y_p[1] > y_p[0] and y_p[1] > y_p[2] ): 
                correct += 1
            elif ( int(y_t) == 2 and y_p[2] > y_p[0] and y_p[2] > y_p[1] ):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    print("My First Model")
    epoch_num=20 # 训练轮次
    batch_size=20 # 每轮训练的样本数量
    train_sample=50000; #样本数量
    input_size = 2 # 输入数据维度
    learning_rate=0.02# 学习率
    
    # 创建模型
    model = MyCossEntropy(input_size)
    # 选择优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    log = []
    # 创建数据集
    train_x,train_y = build_dataSet( train_sample ) 
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss=[]
        # 随机打乱样本
        for batch_index in range(train_sample // batch_size):    
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model.forward(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 梯度归零
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


if __name__ == "__main__":
    main()
    
    