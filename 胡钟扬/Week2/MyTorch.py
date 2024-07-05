import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt





class TorchModel(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(TorchModel,self).__init__()
        self.linear1=nn.Linear(input_size, 10)
        self.linear2=nn.Linear(10, 20)
        self.linear3=nn.Linear(20, output_size)
        self.activation=nn.ReLU()
        self.loss=nn.CrossEntropyLoss()
        
    def forward(self,x,y_true=None):
        x = self.linear1(x)
        x= self.activation(x)
        # print(f'x-linear1 = \n{x}')
        
        x = self.linear2(x)
        x= self.activation(x)
        # print(f'x-linear2 = \n{x}')
        
        x = self.linear3(x)
        x= torch.softmax(x, dim=1)
        
        y_pred=x
        # print(f'y_pred = \n{y_pred}')
         
        # x =nn.Softmax(x)
        if y_true is None:
            return y_pred
        
        else: # 计算损失
            return self.loss(y_pred, y_true)
        
        
        


def build_sample():
    x = np.random.normal(0, 10, 5)
    
    max = np.max(x)
    # for i, n in enumerate(x):
    #     if n==max:
    #         return x,i
    return x, np.where(x == max)[0][0]



def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)



def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # x: 100x5
    # y: 100x1
    
    # print(f'x = \n{x}')
    # print(f'y = \n{y}')
    
    
    y=y.reshape((100,1))
 
    
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测   nx5
        for y_p,y_t in zip(y_pred, y):
            if torch.argmax(y_p)==int(y_t):  # 预测正确
                correct += 1
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
    output_size=5 # 分类的数量
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size,output_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train() # 进入训练阶段
        watch_loss = []
        for batch_index in range(train_sample // batch_size):    
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss    model.forward
            loss.backward()  # 计算梯度       [一个批次对应一组权重，一组权重对应一个模型，这个模型拿来对批次中的每一个模型计算梯度并累加]
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零    [每个批次的梯度累加后只负责更新权重一次]
            loss_item : Number= loss.item() # 返回该批次的平均损失
            watch_loss.append(loss.item())  # item()是Tensor继承_TensorBase类的成员方法
            
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


# 预测
def predict(model_path, input_vec):
    input_size = 5
    output_size=5
    model = TorchModel(input_size, output_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测  nx5
        
    
    # result=torch.argmax(result, dim=1)
    result = torch.softmax(result, dim=1)
        
    
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, torch.argmax(res), res[torch.argmax(res)]))  # 打印结果


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="True"
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.pt", test_vec)