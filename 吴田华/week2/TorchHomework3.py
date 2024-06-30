import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 修改为多分类输出
        self.activation = nn.Softmax(dim=1)  # 使用Softmax进行多分类归一化

    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            loss = nn.CrossEntropyLoss()(y_pred, y)  # 使用交叉熵损失
            return loss
        else:
            return y_pred

def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)  # 获取最大值的索引
    return x, max_index  # 返回样本和对应的类别

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 标签改为长整型

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d 个样本" % test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        _, predicted = torch.max(y_pred, 1)  # 获取预测的类别
        for p, t in zip(predicted, y):
            if p == t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    num_classes = 5  # 五分类
    learning_rate = 0.001

    model = TorchModel(input_size, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d 轮平均 loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), "model.pt")

    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    _, predicted = torch.max(result, 1)  # 获取预测的类别
    for vec, res in zip(input_vec, predicted):
        print("输入：%s, 预测类别：%d" % (vec, res))

if __name__ == "__main__":
    main()
    test_vec = [[0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.1]]
    predict("model.pt", test_vec)