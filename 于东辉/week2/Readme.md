```
import torch  #PyTorch库，用于构建和训练神经网络。
import torch.nn as nn  #PyTorch库中的神经网络模块，包含构建神经网络所需的层和损失函数
import numpy as np  #numpy库，用于高效的数值计算
import random  #python的随机数生成模块
import json  #python的json模块，用于处理json数据格式
import matplotlib.pyplot as plt #绘图模块，数据可视化
import torch.nn.functional as F #softmax作为激活函数
import torch.optim as optim #优化器
import os
'''
基于pytorch进行模型训练，找规律
规律：输入数据是一个5维的向量，判断其中最大值为正标签，其他为负标签
'''

#生成随机五维样本并返回样本和标签
def build_sample():
    x=np.random.random(5)
    max_index=np.argmax(x)#找出最大值的索引
    y=np.zeros(5, dtype=int)  #建立零向量
    y[max_index]=1  #将最大值设置为1
    return x,y

#随机生成批次样本
def build_dataset(total_sample_num):
    X=[]
    Y=[]
    for i in range(total_sample_num): #for循环生成num次数的样本
        x,y=build_sample()
        X.append(x)
        Y.append(y)
        X_app=np.array(X) #将X和Y列表转换为numpy数组，方便后面torch转换
        Y_app=np.array(Y)
    return torch.tensor(X_app,dtype=torch.float32),torch.tensor(Y_app,dtype=torch.float32) #将numpy数组转换为torch形式


#写torch训练模型
class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchModel, self).__init__()
        self.Linear1 = nn.Linear(input_size, 20)  # 更改了层的命名以符合 Python 的命名规范
        self.Linear2 = nn.Linear(20, output_size)  # 确保 output_size 与类别数量相匹配

    def forward(self, x):
        x = F.relu(self.Linear1(x))  # 使用 ReLU 激活函数
        y_preid = self.Linear2(x)  # 第二个线性层，输出 logits
        # 这里不再需要显式调用 log_softmax，因为 nn.CrossEntropyLoss 会内部处理
        return y_preid


def evaluate(model):
    model.eval()  # 将模型设置为评估模式
    test_sample_num = 100  # 假设测试集大小为100
    x, y = build_dataset(test_sample_num)  # 构建测试数据集，您需要实现这个函数

    correct_positive = 0  # 正确预测的正样本数量

    with torch.no_grad():  # 不计算梯度，节省计算资源
        y_pred = model(x)  # 模型预测
        # 将预测结果转换为0和1，其中大于等于0.5的预测结果为1，否则为0

        # 比较预测结果和真实标签
        for y_p, y_t in zip(y_pred, y):
            # 检查预测的最大元素索引是否与真实标签的最大元素索引相同
            # 并且预测和真实标签都是正样本（即1）
            if torch.argmax(y_p) == torch.argmax(y_t):
                correct_positive += 1

    # 计算准确率
    accuracy = correct_positive / test_sample_num
    print("正确预测正样本个数：%d, 准确率：%f" % (correct_positive, accuracy))
    return accuracy


model_save_path = 'D:/Models/iris-predict-module.pt'
def train():
    epoch_num=100
    batch_size=20
    train_sample=4000
    module=TorchModel(5,5)
    loss=nn.CrossEntropyLoss()
    optim_xy=optim.Adam(module.parameters(),lr=0.001) #优化器
    log=[]
    train_X, train_Y = build_dataset(train_sample)
    for epoch  in range(epoch_num):
        module.train()
        watch_loss=[]
        for batch_index in range(train_sample//batch_size):
            x=train_X[batch_index*batch_size:(batch_index+1)*batch_size]
            y=train_Y[batch_index*batch_size:(batch_index+1)*batch_size]
            y_preid=module(x)#计算module的预测值
            loss_index=loss(y_preid,y)
            optim_xy.zero_grad() #将旧的梯度归零
            loss_index.backward()#反向传播
            optim_xy.step()
            watch_loss.append(loss_index.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(module)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 指定模型保存路径

    # 确保目录存在
    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))

    # 保存模型
    torch.save(module.state_dict(), model_save_path)

    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return
# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size, 5)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    input_tensor = torch.tensor(input_vec, dtype=torch.float32)  # 将输入转换为张量

    with torch.no_grad():  # 不计算梯度
        result = model(input_tensor)  # 模型预测，注意这里直接使用模型的 __call__ 方法

    for vec, res in zip(input_vec, result):
        predicted_class = torch.argmax(res).item()  # 获取预测的类别索引
        max_prob, _ = res.max(dim=0)  # 获取最高概率值和对应的索引
        print(f"输入：{vec}, 预测类别：{predicted_class}, 最大概率值：{max_prob.item()}")

if __name__ == "__main__":
            # main()
            train()
            test_vec = [[0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                        [0.94963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
                        [0.78797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
                        [0.79349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]]
            predict(model_save_path, test_vec)
```