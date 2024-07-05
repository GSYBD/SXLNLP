# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 定义模型
class MultiClassificationDemo(nn.Module):
    def __init__(self, input_size):
        super(MultiClassificationDemo, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层, 5分类(out_features:5)
        self.loss = nn.functional.cross_entropy  # 损失函数采用交叉熵, loss为一个函数的指针，尚未传参

    def forward(self, x, y=None):
        y_pred = self.linear(x)  # 计算出模型结果的值
        if y is None:
            return y_pred  # 只输训练数据时，返回线性层输出的预测值
        else:
            return self.loss(y_pred, y)  # 有输入样本标记的正确结果，返回传参后损失函数的值(返回一个tensor)


# 定义生成训练数据的函数
def build_sample():
    x = np.random.random(5)
    # 获取最大值的索引
    max_index = np.argmax(x)
    if max_index == 0:
        return x, 0
    elif max_index == 1:
        return x, 1
    elif max_index == 2:
        return x, 2
    elif max_index == 3:
        return x, 3
    else:
        return x, 4
def build_dataset(create_num):
    X = []
    Y = []
    for i in range(create_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 评估模型 创建test数据集，并计算准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    """
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    """
    input_size = 5  # 输入维数
    learning_rate = 0.001
    epoch_num = 20  # 训练轮数
    train_sample = 5000  # 样本数量
    batch_size = 20  # 每次训练样本个数
    # 建立模型
    model = MultiClassificationDemo(input_size)
    # 创建的是优化器的实例, 其中parameters()为模型实例的方法，返回模型中所有可训练参数的迭代器。Adam优化器将使用这些参数来更新模型的权重和偏差。
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练数据集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model.forward(x, y)  # 计算loss, 等同于 model(x, y)
            loss.backward()  # loss是一个tensor，值为scalar，不需要输入参数矩阵
            optim.step()  #
            optim.zero_grad()
            # tensor.item()方法 只能把只有一个元素的tensor转换为scalar，其他情况请参见.tolist()
            watch_loss.append(loss.item())  # 损失函数值的列表
        print("===============\n第{}轮平均loss:{}".format(epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])  # 记录准确率和损失函数的平均值
    # # 保存模型
    torch.save(model.state_dict(), "model.pt")
    # 画图
    print(log)
    # # 需要学习 matplotlib.pyplot as plt包
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, test_data):
    input_size = 5
    model = MultiClassificationDemo(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())  # 打印参数w和b

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(test_data))  # 模型预测
        print(result)
    for vec, res in zip(test_data, result):
        print("================================================================\n"
            " 输入：%s, 预测类别：%d \n 输出：%s, 实际类别: %s" % (vec, int(torch.argmax(res)), res, np.argmax(vec)))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.09871392],
                [0.89349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894]]
    predict("model.pt", test_vec)
