import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数最大，则为第1类；
                   如果第2个数最大，则为第2类；
                   ...
                   如果第5个数最大，则为第5类。

"""
#搭建一个2层的神经网络模型
class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(TorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1) #w：n * 8
        self.layer2 = nn.Linear(hidden_size1, hidden_size2) # n * 5
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y = None):
        x = self.layer1(x)   #shape: (batch_size, input_size) -> (batch_size, hidden_size1)
        # print(x)
        y_pred = self.layer2(x) #shape: (batch_size, hidden_size1) -> (batch_size, hidden_size2)
        # print(y_pred)
        if y is not None:
            return self.loss(y_pred, y)  # 返回loss值
        else:
            return torch.softmax(y_pred, dim=1)  # 输出预测结果

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第i个数最大，则为第i类
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 找到最大值的索引
    return x, y + 1

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y - 1)
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个样本" % (len(y.numpy())))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        # print(y_pred)
        # print(np.argmax(y_pred))
        # print(np.argmax(y_pred.numpy()))
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            # print(np.argmax(y_p))
            # print(np.argmax(y_p.numpy()))
            # print(int(y_t))
            if np.argmax(y_p.numpy()) == int(y_t):
                correct += 1  #分类正确
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
    hidden_size1 = 8  # 隐藏层1维度
    hidden_size2 = 5  # 隐藏层2维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size, hidden_size1, hidden_size2)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # print(train_x)
    # print(train_y)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            # print(x)
            # print(y)
            target = torch.LongTensor(y.numpy())
            # print(target)
            loss = model(x, target)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    # 画图
    print(log)
    # z = np.array(log)
    # print(z.shape)
    # xx = [l[0] for l in log]
    # print(xx)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    hidden_size1 = 8
    hidden_size2 = 5
    model = TorchModel(input_size, hidden_size1, hidden_size2)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())
    print("====================================================")
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model(torch.FloatTensor(input_vec))  # 模型预测
        # print("===========================================")
        # print(result)
    for vec, res in zip(input_vec, result):
        # print(res)
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, np.argmax(res.numpy()) + 1, np.max(res.numpy())))  # 打印结果


if __name__ == "__main__":
    # main()
    print("\n固定数据测试：")
    test_vec1 = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.pt", test_vec1)
    print("====================================================")
    print("\n随机数据测试：")
    test_vec2 = []
    for i in range(10):
        x = np.random.random(5)
        test_vec2.append(x)
    predict("model.pt", np.array(test_vec2))