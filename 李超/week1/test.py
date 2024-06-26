import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
"""
基于 pytorch 框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x 是一个 10 维向量，如果第 1 个数最大则为一类，如果第 2 个数最大则为二类，以此类推
"""
class MetModel(nn.Module):
    def __init__(self, input_size, num_size):
        super(MetModel,self).__init__()
        #线性层
        self.liner = nn.Linear(input_size, num_size)
        #Softmax归一化函数
        self.activation = torch.sigmoid
        #loss 函数采用交叉熵损失
        self.loss = nn.functional.mse_loss
        # 当输入真实标签，返回 loss 值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # (batch_size, input_size) -> (batch_size, num_size)
        x = self.liner(x)
        # (batch_size, num_size) -> (batch_size, num_size)
        y_pred = self.activation(x)
        # 预测值和真实值计算损失
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred   # 输出预测结果
# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
def build_sample():
    x = np.random.random(10)
    max_index = np.argmax(x)
    return x, [max_index + 1] #类别从 1 开始

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x,y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y) # 标签改为长整型

def evaluate(model):
    model.eval()
    test_total_num = 100
    x, y = build_dataset(test_total_num)
    print("本次预测集中各类别样本数量")
    untion,counts = np.unique(y.numpy(),return_counts=True)
    for category,count in zip(untion,counts):
        print(f"类别:{category},数量:{count}个")
        correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        _, predicted = torch.max(y_pred, 1) # 获取预测的类别
        for y_p,y_t in zip(predicted, y):
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)
def main():
    # 配置参数
    epoch_num = 20  #训练轮数
    batch_size = 20 #每次训练样本个数
    train_sample = 5000 #每轮训练总共训练的样本总数
    input_size = 10 #输入向量维度
    num_size = 10 # 类别数
    learning_rate = 0.001 #学习率
    # 建立模型
    model = MetModel(input_size,num_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(),lr = learning_rate)
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
            loss.backward()    # 计算梯度
            optim.step()       # 更新权重
            optim.zero_grad() # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d 轮平均 loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc,float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return
# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 10
    num_size = 10
    model = MetModel(input_size, num_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        _, predicted = torch.max(result, 1)  # 获取预测的类别
    for vec, res in zip(input_vec, predicted):
        print("输入：%s, 预测类别：%d" % (vec, res + 1))  # 打印结果
if __name__=="__main__":
    main()
    test_vec =[[1.07889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843, 0.07889087, 0.07889088,  0.07889089, 0.07889090, 0.77889091],
               [2.94963533, 0.5524256, 0.95758807, 0.95520434,0.84890681, 0.84890682, 0.84890683, 0.84890684, 0.84890685, 0.95890686],
               [3.78797868, 0.67482528, 0.13625847, 0.34675372,0.19871392, 0.19871393, 0.19871394, 0.19871395, 0.07889099, 0.29871397],
               [4.79349776, 0.59416669, 0.92579291, 0.41567412,0.1358894,0.1358895, 0.1358896, 0.1358897, 0.1358898, 0.89889090]]
    predict("model.pth", test_vec)