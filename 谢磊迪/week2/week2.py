import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
"""
    pytorch 框架 搞一个模型  
    实现一个基于交叉熵的多分类任务，任务可以自拟。
    规律：x是一个10维向量，如果第1个数>第5个数 为[1,0,0]  ;如果第1个数>第5个数 且  第2个数<第5个数 为[0,1,0]  ;
                        如果第1个数<=第5个数 为[0,0,1]
"""
def log_data():
    a = np.random.rand(10)
    if a[0]>a[4]:
        return a,[1,0,0]
    elif a[0]>a[4] and a[1]<a[4]:
        return a,[0,1,0]
    elif a[0]<=a[4]:
        return a,[0,0,1]
def make_datas(train_sample):
    li_x = []
    li_y = []
    for i in range(train_sample):
        x,y = log_data()
        li_x.append(x)
        li_y.append(y)
    return torch.Tensor(li_x),torch.Tensor(li_y)




class Net10_3class(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear  = nn.Linear(input_dim,5)
        self.layer2 = nn.Linear(5, 3)
    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x


def evaluate(model):
    model.eval()
    with torch.no_grad():
        test_sample = 500
        test_x, test_y = make_datas(test_sample)
        output = model.forward(test_x)
        correct = 0
        wrong = 0
        for y_p, y_t in zip(output, test_y):
            predicted_index = torch.argmax(y_p).item()
            predicted_index1 = torch.argmax(y_t).item()
            if predicted_index == predicted_index1:
                correct += 1
            else:
                wrong += 1
        print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))





def main():
    epoch_num = 10  # 训练轮数
    learning_rate = 0.01
    train_sample = 5000
    batch_size = 500
    model = Net10_3class(10)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #create tran_datas
    tran_x,tran_y= make_datas(train_sample)
    for epoch in range(epoch_num):
        model.train()
        loss_li = []
        for iteration in range(train_sample//batch_size):
            batch_x = tran_x[iteration*batch_size:(iteration+1)*batch_size]
            batch_y = tran_y[iteration*batch_size:(iteration+1)*batch_size]
            # optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            # 反向传播和优化
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 清空梯度
            loss_li.append(loss.item())
        print(f'第{epoch}轮',f'loss:{np.mean(loss_li)}')
        evaluate(model)
    print('aaaaaa')
    torch.save(model.state_dict(), "model1.pt")

# 加载整个模型


# 使用模型进行预测的步骤与上面相同
# ...
def  predict(model_dict):
    model = Net10_3class(10)
    model.load_state_dict(torch.load(model_dict))
    model.eval()
    test_x, test_y = make_datas(10)
    # print(test_x)
    with torch.no_grad():
        predictions = torch.sigmoid(model(test_x))
        for i, j, z in zip(test_x, predictions, test_y):
            print(i, '\n', j, z)


if __name__ == '__main__':
    # main()
    model = Net10_3class(10)
    predict("model1.pt")
















