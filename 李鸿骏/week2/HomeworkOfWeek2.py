import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


# 任务：x是长度为5的向量，若第一个数最大，则标签类型为0,若第二个数最大，则标签为1，以此类推，

# 自定义模型
class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = torch.softmax
        self.loss_func = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x, dim=1)
        if y is not None:
            return self.loss_func(y_pred, y)
        else:
            result = []
            for y_p in y_pred:
                result.append(torch.argmax(y_p))
            return result, y_pred


# 生成样本的方法
def generate_samples(train_size):
    samples = []
    labels = []
    for i in range(train_size):
        sample = np.random.random(5)
        max_index = np.argmax(sample)
        samples.append(sample)
        labels.append(max_index)
    return torch.FloatTensor(samples), torch.LongTensor(labels)


# 评估模型正确率
def evaluate(model):
    model.eval()
    # 生成测试数据
    x_test, y_test = generate_samples(100)
    result, y_pred = model(x_test)
    correct_num = 0
    error_num = 0
    for i in range(len(x_test)):
        # print(f"======================样本:{x_test[i]}")
        # print(f"======================预测结果:{y_pred[i]}")
        # print(f"======================预测标签:{result[i]},真实标签:{y_test[i]}")
        if torch.argmax(y_pred[i]) == y_test[i]:
            correct_num += 1
        else:
            error_num += 1

    print(f"预测正确数：{correct_num},预测错误数：{error_num},预测正确率：{correct_num / (correct_num + error_num)}")
    return correct_num / (correct_num + error_num)


# 训练
def main():
    # 训练数据样本总数
    train_size = 1000
    # 生成训练数据
    x_train, y_train = generate_samples(train_size)
    # 训练轮数
    epoch_nums = 1000
    # # 输入向量的维度
    input_size = 5
    # 输出向量的维度
    output_size = 5
    # 学习率
    lr = 0.001
    # 每次训练样本数
    batch_size = 20
    # 创建模型实例
    model = MyModel(input_size, output_size)
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # 初始化日志记录
    log = []

    # 开始训练，轮数100,每轮
    for epoch in range(epoch_nums):
        model.train()
        # 初始化损失值记录日志
        loss_log = []
        # 初始化损失值观察
        for i in range(train_size // batch_size):
            # 计算损失值
            loss = model(x_train[i * batch_size: (i + 1) * batch_size], y_train[i * batch_size: (i + 1) * batch_size])
            # 计算梯度
            loss.backward()
            # 更新权重
            optimizer.step()
            # 梯度归零
            optimizer.zero_grad()
            # 记录损失值
            loss_log.append(loss.item())
        # 测试本轮训练结果
        acc = evaluate(model)
        # 计算并打印当前轮的平均损失值
        loss_log.append(loss.item())
        mean_loss = np.mean(loss_log)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, mean_loss))
        log.append([acc, float(np.mean(loss_log))])
        # # 正确率大于98%，退出训练
        # if acc > 0.98:
        #     break
    # 训练结束，保存模型
    torch.save(model.state_dict(), "model.pt")

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    output_size = 5
    model = MyModel(input_size, output_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result, y_pred = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res, y in zip(input_vec, result, y_pred):
        print("输入：%s, 预测类别：%d, 概率为：%f" % (vec, res, torch.max(y)))  # 打印结果


if __name__ == "__main__":
     # main()
    test_vec = [[2.17889086,7.15229675,3.31082123,10.73504317,6.18920843],
                [1.94963533,3.5524256,3.95758807,2.95520434,0.84890681],
                [1.78797868,5.67482528,0.13625847,1.34675372,6.19871392],
                [0.79349776,0.99416669,0.92579291,0.41567412,0.1358894]]
    predict("model.pt", test_vec)
