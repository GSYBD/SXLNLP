import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

"""
一个分类任务，那个数字大就返回那个数字的索引
"""

#1.生成数据
def build_sample():
    x = np.random.randn(6)
    return x, np.argmax(x)

def build_dataset(sample_total):
    X = []
    Y = []
    for _ in range(sample_total):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 2. 构建模型
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, input_size) # 全连接层
        self.loss = nn.CrossEntropyLoss() # 损失函数

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if (y is not None):
            return self.loss(y_pred, y)
        else:
            return y_pred

# 3.没跑一轮的验证
def evaluate(model:TorchModel):
    # 生成数据
    eval_sample_total = 100
    x, y = build_dataset(eval_sample_total)
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        right, wrong = 0, 0
        for y_p, y_t in zip(y_pred, y):
            y_p = torch.argmax(y_p)
            if y_p == y_t:
                right += 1
            else:
                wrong += 1
        print("本次预测一共预测正确%d个,正确率%f" % (right, right/(right + wrong)))
    return right/(right + wrong)

# 4.预测
def predict(model_path,predict_data,input_size):
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        y_pred = model(predict_data)
    for input_data, y_p in zip(predict_data, y_pred):
        print("本次预测:%s,正确结果为%f,预测结果为%f" % (input_data, torch.argmax(input_data), torch.argmax(y_p)))

# 5. 训练函数
def main():
    epoch_num = 20
    input_size = 6
    sample_total = 5000
    learning_rate = 0.01
    batch_size = 20
    model = TorchModel(input_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    X, Y = build_dataset(sample_total)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(sample_total // batch_size):
            train_x = X[batch_index * batch_size: (batch_index + 1) * batch_size]
            train_y = Y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(train_x, train_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        print("第%d轮,loss=%f" % (epoch, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append((acc, np.mean(watch_loss)))
    # print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label = 'acc')
    plt.plot(range(len(log)), [l[1] for l in log], label = 'loss')
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), 'modelc.pth')

if __name__ == '__main__':
   # main()

   res = build_dataset(10)
   predict('modelc.pth', res[0], 6)