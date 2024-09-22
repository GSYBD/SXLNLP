# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


"""
基于pytorch框架,编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律:x 是一个5维向量,如果向量中第2个数的平方大于第3个数的平方,则为正样本,否则为负样本
判断:输入x 维度为5,输出x 维度至少为2, 这里先设置成5
"""
def build_sample():
    """_summary_
    构造样本判断条件
    向量中第2个数的平方大于第3个数的平方,则为正样本,否则为负样本
    Returns:
        _type_: _description_
        [0.60656849 0.94246913 0.60026904 0.83533301 0.0852189 ] 1*5 维度的向量
    """
    x = np.random.random(5)
    if x[1]**2 > x[2]**2:
        return x,1
    else:
        return x,0

def build_dataset(sample_qty):
    """_summary_
    输入一个数据集的目标样本总数,执行qty次,
    Args:
        sample_qty (_type_): _description_

    Returns:
        _type_: _description_
        获取每次生成的随机数据以及其判定结果,
    """
    X=[]
    Y=[]
    for i in range(sample_qty):
        x,y = build_sample()
        X.append(x)
        Y.append([y]) 
    return torch.FloatTensor(X),torch.FloatTensor(Y) # 将列表转化为张量

def evaluate(model):
    """_summary_
    传入模型,评估模型的准确率
    Args:
        model (_type_): _description_
        X: n维张量,这里是五维,
    """
    model.eval()
    test_sample_qty = 200
    # 构造出qty 长度的数据集 X,Y
    # X = [[0.60656849 0.94246913 0.60026904 0.83533301 0.0852189 ],[0.60656849 0.94246913 0.60026904 0.83533301 0.0852189 ],...]
    # Y = [[1],[0],[1],...]
    X,Y = build_dataset(test_sample_qty)
    print(f"第n次正样本数:{sum(Y)},负样本数:{test_sample_qty-sum(Y)}")
    correct,wrong = 0,0
    with torch.no_grad(): # 取消梯度函数的计算
        y_pred= model(X) # 获取预测值,Vect 为构造n次样本后的向量列表,被转化成的张量
        for y_p,y_t in zip(y_pred,Y): # 预测值,和真实值进行比较
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
        super().__init__() # 这里等同于 super(nn.Module,self).__init__()
        self.linear = nn.Linear(input_size,5) # 训练模型整体可以看作是一个全连接层,即线性层,定义好模型输入的尺寸为input_size,模型的输出层尺寸为1
        self.loss = nn.functional.mse_loss # 选择均方差公式来作为损失函数, 准确率50左右
        self.activation = torch.sigmoid # 选择sigmoid 作为归一化函数,激活线性层,目的是为了使偏函数梯度下降非线性
        # self.loss = nn.functional.cross_entropy
        
    def forward(self,x,y=None):
        """_summary_
        forward the input data to train
        当输入的标签为真实标签,那么就对模型进行训练,返回本次训练的loss 值
        当输入的标签为空,那么就使用模型,对输入值进行计算,返回预测值
        无论返回loss值,或者是预测值,都需要对预测值进行计算
        Args:
            x (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.
        """
        item = self.linear(x)
        y_pred = self.activation(item) # 这里的sigmoid是一个函数,直接调用
        if y is not None:
            # return self.loss(input=y,target=y_pred) # 错误写法,对的应该反过来
            # input 是y,是input的x 经过激活后的数据,而y是目标值
            return self.loss(y_pred,y) 
        else:
            return y_pred
        

def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.01
    
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
        