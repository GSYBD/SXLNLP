# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：使用RNN做一个多分类任务，找一个字符串，这个字符串在第几个坐标就属于第几类。多分类交叉熵。
"""

class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TorchModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    def forward(self, x, y=None):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        if y is not None:
            return self.loss(out, y)  # 预测值和真实值计算损失
        else:
            return out  # 输出预测结果

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个字符串，如果字符串中包含"你"，则"你"在字符串中的位置决定类别
def build_sample(max_length=10):
    chars = "abcdefghijklmnopqrstuvwxyz你"
    length = random.randint(1, max_length)
    s = "".join(random.choice(chars) for _ in range(length))
    # 确保字符串中包含"你"
    if "你" not in s:
        s = s[:length//2] + "你" + s[length//2+1:]
    label = s.index("你")
    return s, label

# 将字符串转换为one-hot编码，并进行填充
def string_to_one_hot(s, chars, max_length):
    char_to_index = {char: i for i, char in enumerate(chars)}
    one_hot = np.zeros((max_length, len(chars)))
    for i, char in enumerate(s):
        one_hot[i, char_to_index[char]] = 1
    return one_hot

# 随机生成一批样本
def build_dataset(total_sample_num, max_length=10):
    chars = "abcdefghijklmnopqrstuvwxyz你"
    X = []
    Y = []
    for i in range(total_sample_num):
        s, y = build_sample(max_length)
        x = string_to_one_hot(s, chars, max_length)
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, max_length=10):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num, max_length)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        _, predicted = torch.max(y_pred, 1)
        for y_p, y_t in zip(predicted, y):  # 与真实标签进行对比
            if y_p == y_t:
                correct += 1  # 预测正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 27  # 输入向量维度（26个字母 + "你"）
    hidden_size = 50  # RNN隐藏层维度
    num_classes = 10  # 类别数
    learning_rate = 0.001  # 学习率
    max_length = 10  # 字符串最大长度

    # 建立模型
    model = TorchModel(input_size, hidden_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample, max_length)
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
        acc = evaluate(model, max_length)  # 测试本轮模型结果
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

# 使用训练好的模型做预测
def predict(model_path, input_vecs, input_strs, max_length=10):
    input_size = 27
    hidden_size = 50
    num_classes = 10
    model = TorchModel(input_size, hidden_size, num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        for input_vec, input_str in zip(input_vecs, input_strs):
            input_tensor = torch.FloatTensor(input_vec)  # 直接将NumPy数组转换为tensor
            result = model(input_tensor.unsqueeze(0))  # 在数据前添加一个batch维度
            _, predicted_class = torch.max(result, 1)
            print("输入：%s, 预测类别：%d" % (input_str, predicted_class.item()))  # 打印结果


if __name__ == "__main__":
    # main()
    test_vec = [
        string_to_one_hot("abcd你ef", "abcdefghijklmnopqrstuvwxyz你", 10),
        string_to_one_hot("ghi你jkl", "abcdefghijklmnopqrstuvwxyz你", 10),
        string_to_one_hot("mnopqr你", "abcdefghijklmnopqrstuvwxyz你", 10),
        string_to_one_hot("s你tuvwx", "abcdefghijklmnopqrstuvwxyz你", 10)
    ]
    test_strs = ["abcd你ef", "ghi你jkl", "mnopqr你", "stu你vwx"]
    predict("model.pt", test_vec, test_strs)
