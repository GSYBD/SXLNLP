# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

'''
第二周作业
DongKaiwei
基于交叉熵的多分类任务

问题说明：
使用三个函数: y = x, y = 2 * e**x - 2, y = ln(x + 1) / 2
将平面上单位矩形区域(0, 1) * (0, 1)分为4个子区域, 自下而上分别为区域 0, 1, 2, 3
已知一个点的坐标为(x, y)，根据判断该点属于哪个子区域
'''

# 生成一个随机样本
def sample_generator():
    x = np.random.uniform(0, 1, 2) # 生成2个[0, 1]之间的随机数
    # 根据要求对x进行分类，末位为子区域序号
    y = _getArea(x)
    return x, y


# 获取二维点的子区域序号
def _getArea(x: list[2]):
    if x[1] <= np.log(x[0] + 1) / 2:
        return 0
    elif x[1] <= x[0]:
        return 1
    elif x[1] <= 2 * np.exp(x[0]) - 2:
        return 2
    else:
        return 3


# 生成样本库
def build_dataset(total_sample_num: int):
    X = [] # 输入为二维点
    Y = [] # 输出为子区域序号
    for i in range(total_sample_num):
        x, y = sample_generator()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(2, 5) # 第一层线性函数
        self.linear2 = nn.Linear(5, 4) # 第二层线性函数
        self.activation = nn.Softmax(dim=1) # 激活函数采用softmax
        self.loss = nn.functional.cross_entropy # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear1(x)
        x = self.linear2(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y) # 预测值和真实值计算损失
        else:
            return y_pred # 输出预测结果


# 测试代码，用于检测每轮模型的准确率
def evaluate(model: Net):
    model.eval() # 设置模型为评估模式
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1  # 负样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

'''测试代码
def test_Net():
    testX, testY = build_dataset(3)
    print(testX)
    print(testY)
    model = Net(2)
    model_w = model.state_dict()["linear1.weight"].numpy()
    model_b = model.state_dict()["linear1.bias"].numpy()
    print(model_w, "torch w1 权重")
    print(model_b, "torch b1 权重")
'''

def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = Net()
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
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "Week2_model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    model = Net()
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    
    return result

def do_predict():
    test_num = 10
    test_x, test_y = build_dataset(test_num) 
    result = predict("Week2_model.pt", test_x)
    index = 1
    for vec, res, y_true in zip(test_x, result, test_y):
        print(f'输入：{vec}, 预测类别：{int(torch.argmax(res))}, 实际类别: {y_true}, 概率值：{res}')  # 打印结果
        
        color = 'green'
        result_s = 'T'
        if y_true != int(torch.argmax(res)):
            color = 'red'
            result_s = 'F'
        plt.scatter(vec[0], vec[1], c=color, s=20)
        plt.annotate(f'{index}: {result_s}', (vec[0], vec[1]))
        index += 1
    # 绘图
    x = np.arange(0, 1, 0.02)
    y1 = 2 * np.exp(x) - 2
    y2 = x
    y3 = np.log(x + 1) / 2
    plt.plot(x, y1, c='blue',label='e**x - 1')
    plt.plot(x, y2, c='blue')
    plt.plot(x, y3, c='blue')
    plt.ylim(0, 1)
    plt.show()
    
        
if __name__ == "__main__":
    main()
    do_predict()
    