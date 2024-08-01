import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
"""
    实现一个基于交叉熵的多分类任务，任务可以自拟。
    规律：x是一个10维向量，如果第1个数大于第5个数为[1,0,0] ;如果第1个数等于第5个数为[0,1,0];如果第1个数小于第5个数1为[0,0,1]
"""

class TorchModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear  = nn.Linear(input_dim,5)
        self.layer2 = nn.Linear(5, 3)
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵
    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x) #激活函数用relu
        y = self.layer2(x)
        return y

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个10维向量，如果第1个数大于第5个数为[1,0,0] ;如果第1个数等于第5个数为[0,1,0];如果第1个数小于第5个数1为[0,0,1]
def build_sample():
    x = np.random.rand(10)
    if x[0] > x[4]:
        return x, [1, 0, 0]
    elif x[0] == x[4]:
        return x, [0, 1, 0]
    elif x[0] < x[4]:
        return x, [0, 0, 1]


# 随机生成一批样本
# 样本均匀生成
def build_dataset(total_sample_num):
    tx = []
    ty = []
    for i in range(total_sample_num):
        x, y = build_sample()
        tx.append(x)
        ty.append(y)
    return torch.Tensor(np.array(tx)), torch.Tensor(np.array(ty))



def evaluate(model):
    model.eval()
    with torch.no_grad():
        test_sample = 1000
        test_x, test_y = build_dataset(test_sample)
        output = model.forward(test_x)
        correct = 0
        wrong = 0
        for y_p, y_t in zip(output, test_y):
            predicted_index = torch.argmax(y_p).item()
            predicted_index1 = torch.argmax(y_t).item()
            if predicted_index == predicted_index1:
                correct += 1
            else:
                wrong += 1
        print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
        return correct / (correct + wrong)




def main():
    epoch_num = 10  # 训练轮数
    learning_rate = 0.01 # 学习率
    train_sample = 10000 # 每轮训练总共训练的样本总数
    batch_size = 1000 # 每次训练样本个数
    model = TorchModel(10) # 建立模型
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #创建训练集，正常任务是读取训练集
    train_x,train_y= build_dataset(train_sample)
    log = []
    for epoch in range(epoch_num):
        model.train()
        loss_li = []
        for iteration in range(train_sample//batch_size):
            batch_x = train_x[iteration*batch_size:(iteration+1)*batch_size]
            batch_y = train_y[iteration*batch_size:(iteration+1)*batch_size]
            output = model(batch_x)
            loss = criterion(output, batch_y) # 计算loss
            # 反向传播和优化
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 清空梯度
            loss_li.append(loss.item())
        print(f'第{epoch+1}轮',f'loss:{np.mean(loss_li)}')
        acc = evaluate(model)
        log.append([acc, float(np.mean(loss_li))])
    print('==========================================================================================================')
    torch.save(model.state_dict(), "model.pt")
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

# 加载整个模型


# 使用模型进行预测的步骤与上面相同
# ...
def  predict(model_dict):
    model = TorchModel(10)
    model.load_state_dict(torch.load(model_dict))
    model.eval()
    test_x, test_y = build_dataset(10)
    with torch.no_grad():
        predictions = torch.sigmoid(model(test_x))
        for i, j, z in zip(test_x, predictions, test_y):
            print(i, '\n', j,'\n', z)


if __name__ == '__main__':
    main()
    model = TorchModel(10)
    predict("model.pt")