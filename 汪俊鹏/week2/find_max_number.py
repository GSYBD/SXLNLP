import numpy as np
import torch
import torch.nn as nn
import random
import json
import matplotlib.pyplot as plt

def build_sample():##数据规则
    x=np.random.random(5)
    y=np.zeros(5)
    y[np.argmax(x)]=1
    return x,y

def build_dataset(total_samlple_num):##生成数据
    X=[]
    Y=[]
    for i in range(total_samlple_num):
        x,y=build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)),torch.FloatTensor(np.array(Y))

class TorchModel(nn.Module):
    def __init__(self,input_size):
        super(TorchModel,self).__init__()
        self.linear=nn.Linear(input_size,5)
        self.activation=torch.sigmoid
        self.loss=nn.CrossEntropyLoss()
    def forward(self,x,y=None):
        x=self.linear(x)
        y_pred=self.activation(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    a,b,c,d,e=0,0,0,0,0
    for i in range(len(y)):
        if y[i].argmax()==0:
            a+=1
        elif y[i].argmax()==1:
            b+=1
        elif y[i].argmax()==2:
            c+=1
        elif y[i].argmax()==3:
            d+=1
        else:
            e+=1
    print("本次预测集中共有%d个一类样本，%d个二类样本，%d个三类样本，%d个四类样本，%d个五类样本" % (a,b,c,d,e))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if int(y_p.argmax())==int(y_t.argmax()):
                correct += 1  # 样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)
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
    torch.save(model.state_dict(), "max_number.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, int(res.argmax())+1,res.max()))  # 打印结果



if __name__ == "__main__":
     main()
     # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
     #        [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
     #        [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
     #        [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
     # predict("model.pt",test_vec)
