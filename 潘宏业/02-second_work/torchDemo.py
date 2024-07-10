import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

"""
实现功能：随机生成五个数，返回数字最大的下标
两个线性层和两个激活层，交叉熵损失函数
"""


class TorchModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x_linear1 = self.linear1(x)
        x_act = self.act(x_linear1)
        x_linear2 = self.linear2(x_act)
        y_pred = self.act(x_linear2)
        if y is None:
            return y_pred
        else:
            return self.loss(y_pred, y)

def data():
    x = np.random.random(5)
    y = np.argmax(x)
    return x, y

def dataset(datanum):
    X = []
    Y = []
    for i in range(datanum):
        x, y = data()
        X.append(x)
        Y.append(y)
    return torch.Tensor(X), torch.LongTensor(Y)

def evaluate(model):
    model.eval()
    eval_num = 200
    x, y = dataset(eval_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if int(y_t) == int(torch.argmax(y_p)):
                correct += 1
            else:
                wrong += 1
    print("预测正确概率为%f，预测正确个数为%d" % (correct/(correct+wrong), correct))
    return correct/(correct+wrong)


def main():
    datanum = 500
    X, Y = dataset(datanum)
    batch_size = 64
    model = TorchModel(5, 100, 5)
    epoch_num = 200
    lr = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(datanum // batch_size):
            optimizer.zero_grad()
            x = X[batch_size*batch : batch_size*(batch + 1)]
            y = Y[batch_size*batch : batch_size*(batch + 1)]
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        print("第%d轮平均loss为%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([np.mean(watch_loss), acc])
    torch.save(model.state_dict(), "model.pth")
    plt.plot(range(len(log)), [l[0] for l in log], label='loss')
    plt.plot(range(len(log)), [l[1] for l in log], label='acc')
    plt.show()    

def predict(model_path, input_data):
    model = TorchModel(5, 100, 5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result = model(torch.Tensor(input_data))
    for vec, res in zip(input_data, result):
        print("输入数据为%s，预测类别为%d，预测概率值为%f" % (vec, int(torch.argmax(res)), float(torch.max(res))))


if __name__ == "__main__":
    # main()
    input_vec = np.random.random((10, 5))
    predict("./model.pth", input_vec)