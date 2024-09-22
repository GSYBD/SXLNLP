# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
cross_entropy 期望输入的是二维的张量,形状为(N,C)
N 为批次大小,C为类别数
目标张量y应该是一维的,形状为(N,),其中包含类别的索引
sigmoid 通常用于二分类或多标签分类任务
cross_entropy 通常用于多分类任务并期望使用softmax激活函数
计算损失的时候,y_pred 应该是未激活的logits,用于softmax 之前的输出
先确定损失函数,然后再选择合适的激活函数,有些函数的激活函数不需要再手写
"""
"""
基于pytorch框架,编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律:x 是一个5维向量,如果向量中第2个数的平方大于第3个数的平方,则为正样本,否则为负样本
判断:输入x 维度为5,输出x 维度至少为2, 这里先设置成5
"""
def build_sample():
    x = np.random.random(5)
    if x[1]**2 > x[2]**2:
        return x,1
    else:
        return x,0

def build_dataset(sample_qty):
    X=[]
    Y=[]
    for i in range(sample_qty):
        x,y = build_sample()
        X.append(x)
        Y.append(y) 
    return torch.FloatTensor(X),torch.LongTensor(Y) # 将列表转化为张量

def evaluate(model):
    model.eval()
    test_sample_qty = 200
    X,Y = build_dataset(test_sample_qty)
    print(f"第n次正样本数:{sum(Y)},负样本数:{test_sample_qty-sum(Y)}")
    correct,wrong = 0,0
    with torch.no_grad(): 
        y_pred= model(X) 
        for y_p,y_t in zip(y_pred,Y):
            if y_p[1]**2 >y_p[2]**2 and int(y_t) == 1:
                correct += 1
            elif y_p[1]**2 <=y_p[2]**2 and int(y_t) == 0:
                correct += 1
            else:
                wrong+= 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)
        

class TorchModel(nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size,5)
        # self.loss = nn.functional.cross_entropy
        # self.activation = torch.sigmoid 
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self,x,y=None):

        item = self.linear(x)
        # y_pred = self.activation(item) 
        if y is not None:
            return self.loss(item,y) 
        else:
            return item
        

def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 1
    
    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    log = []
    train_x,train_y = build_dataset(train_sample)
    
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index *batch_size:(batch_index+1) *batch_size]
            y = train_y[batch_index *batch_size:(batch_index+1) *batch_size]
            loss = model(x,y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model_X.pt")
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
        