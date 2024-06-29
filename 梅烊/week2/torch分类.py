import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import os


class TorchClassModel(nn.Module):
    def __init__(self, input_size,out_size):
        super(TorchClassModel,self).__init__()
        self.linear = nn.Linear(input_size,out_features=out_size)
        self.activation = torch.softmax
        # 指定损失函数
        self.loss =nn.functional.cross_entropy

    def forward(self, input, y=None):
        input = self.linear(input)
        # softmax.dim=1 在行上进行softmax
        y_pred = self.activation(input,1)  
        if y is None:
            # y is none ，进行预测，则返回预测值，不计算loss值
            return y_pred
        else:
            # y is not none ，进行训练，则计算loss值
            return self.loss(y_pred,y)  

# 生成训练样本
def build_sample():
    x= np.random.random(5)
    if np.sum(x) < 2:
        return x,[1,0,0]
    elif np.sum(x) >= 2 and np.sum(x) < 2.5:
        return x,[0,1,0]
    else:
        return x,[0,0,1]
    
# 生成一批样本
def build_dense(total_num):
    X=[]
    Y=[]
    for i in range(total_num):
        x,y = build_sample()
        X.append(x)
        Y.append(y)
        # print(X,Y)    
    # print(torch.FloatTensor(np.array(Y)).size(-1))
    return torch.FloatTensor(np.array(X)),torch.FloatTensor(np.array(Y))

# 模型评估
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x,y = build_dense(test_sample_num)
    
    correct_i, wrong_i = 0, 0

    # 开始评估
    with torch.no_grad():
        y_pred = model(x)
        
        # 逐行对比预测值
        for y_t,y_p in zip(y,y_pred):
            if (y_t == y).all():
                correct_i += 1
            else:
                wrong_i += 1
    print("模型评估结果=======================，正确预测个数：%d, 正确率为%f",correct_i , correct_i/(correct_i + wrong_i))
        
def predict(model, ds):
    model.eval()
    with torch.no_grad():
        y = model.forward(ds)
    for ds_vec, y in zip(ds, y):
        print("输入：%f, 预测类别：%d, 概率值：%f" %(np.sum(np.array(ds_vec)), np.argmax(np.array(y)), np.array(y).max()))

def main():
    # 配置参数
    epoch = 100 #训练轮数
    batch_size = 10 #小批量size
    train_sample_size =5000  #训练样本数量
    input_size = 5  #输入向量维度
    learning_rate = 0.01 #学习率

    # 模型测试集
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
            [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
            [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
            [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]

    # 创建训练集
    train_x, train_y = build_dense(train_sample_size)

    # 建立模型
    model = TorchClassModel(input_size=input_size,out_size=train_y.size(-1))
    
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    log = []

    # 开始训练
    for i in range(epoch):
        # 开启模型的训练模式
        model.train()
        
        # 累计损失
        total_loss = []
        # 开始小批量训练 
        for batch_index in range(train_sample_size//batch_size) :
            x = train_x[batch_size * batch_index : batch_size *(batch_index+1) ] 
            y = train_y[batch_size * batch_index : batch_size *(batch_index+1) ] 
            # 计算损失
            loss = model.forward(x,y)
            # 计算梯度
            loss.backward()
            # 权重更新
            optim.step()
            # 权重置零
            optim.zero_grad()
            # 记录每个小批量的loss值
            total_loss.append(loss.item())

        log.append(np.mean(total_loss))
        
        # 打印每一轮训练的loss平均值
        # print("=========\n第%d轮平均loss:%f" % (i + 1, np.mean(total_loss)))

    # plt.plot(log)
    # plt.show()

    # 使用模型进行预测
    predict(model, torch.tensor(test_vec))


if __name__ == "__main__":
    # os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'
    main()

    # print(build_dense(10))
    # x,y = build_sample()
    # print(x,y)
    # print(np.sum([0.6980528  ,0.08761461 ,0.18281877 ,0.72726902 ,0.75050663]))