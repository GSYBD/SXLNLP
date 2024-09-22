import torch
import torch.nn as nn
import numpy as np

# import torch.nn.functional as Func

class TorchModel(nn.Module):
    def __init__(self,input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果

    # 生成一个样本
    # 随机生成一个5维向量，如果第一个值最大则为第一类，以此类推


def build_sample():
    x = np.random.random(5)
    y_t = np.argmax(x)
    return x,y_t


def build_dataset(total_sample_num):
    x_data = []
    y_data = []
    for i in range(total_sample_num):
        x, y = build_sample()
        x_data.append(x)
        y_data.append(y)
    # return torch.FloatTensor(x_data), torch.LongTensor(y_data)
    return torch.FloatTensor(np.array(x_data)), torch.LongTensor(np.array(y_data))

def evaluate(model):
    model.eval()
    test_sample_num = 10
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():        # 不计算梯度
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if  torch.argmax(y_p) == int(y_t):
                correct += 1         # 预测正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 构建模型
def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)  # 定义模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 优化器采用Adam
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练模型
    # log = []
    for epoch in range(epoch_num):
        model.train()
        # 训练集分批训练
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一批训练数据
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            # print(x,y)
            loss = model(x, y)  # 计算损失
            loss.backward()  # 反向传播
            optim.step()  # 更新参数
            optim.zero_grad()  # 清空梯度
            watch_loss.append(loss.item())  # 记录每批的损失
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        # log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    # torch.save(model.state_dict(), "CETest1.pt")
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        # print(result)
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.pt", test_vec)
    # aaa = build_dataset(5)
    # print(aaa)

