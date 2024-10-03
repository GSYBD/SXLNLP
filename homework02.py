import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

# 基于pytorch框架编写训练模型
# 实现一个自行找规律的(机器学习)任务
# 规律：x是一个5维向量，向量中哪个标量最大就输出哪一维下标
# 等索引完，就差不多了
# 这就好了，这个代码的后缀需要是py，刚才是没有
# 好的
# 还有个就是没有用
# py36                  *  D:\360Downloads\Anaconda\envs\py36
# 这个py环境，这个环境中安装的torch，但是你用其他的环境，就不一定又这个torch了
# 好，那之后每次都要在cmd中激活吗
# 在cmd中激活的目的是为了安装torch，你激活了哪个环境，然后在该环境下安装包，包就会被安装到这个环境下
# 好的好的，懂了
# 你可以创建一个文件夹，每次写代码的时候，在这个文件夹下创建py文件，就不用每次都去更改py的解释器了
# """


class MultiClassficationModel(nn.Module):
    def __init__(self, input_size):
        super(MultiClassficationModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        self.loss = nn.CrossEntropyLoss()

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred




# 生成一个样本，样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，根据每个向量中最大的标量同一下标构建Y
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x) #获取最大值的索引
    return x, max_index



# 随机生成一批样本
def build_dataest(total_sample_size):
    X = []
    Y = []
    for i in range(total_sample_size):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


#测试代码
#用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_sum = 100
    x, y = build_dataest(test_sample_sum)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred,y):
            if torch.argmax(y_p) == int(y_t):
                correct +=1
            else:
                wrong +=1
    print("正确预测个数：%d, 正确率：%f" %(correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    train_sample = 5000
    epoch_num = 200
    batch_size = 20
    input_size = 5
    learning_rate = 0.001
    #建立模型
    model = MultiClassficationModel(input_size)
    #选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    #创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataest(train_sample)
    #训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  #计算loss
            loss.backward()  #计算梯度
            optim.step()   #更新权重
            optim.zero_grad()  #梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  #测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    #保存模型
    torch.save(model.state_dict(), "model.pt")
    #画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    return

#使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = MultiClassficationModel(input_size)
    model.load_state_dict(torch.load(model_path))  #加载训练好的权重
    print(model.state_dict())

    model.eval()  #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  #模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res))  #打印结果


if __name__ == "__main__":
    #main()
    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.94963533, 0.5524256, 0.9578807, 0.95520434,0.84890681],
                [0.7897868, 0.67482528, 0.13625847, 0.24675372,0.09871392],
                [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.pt", test_vec)