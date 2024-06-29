import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

'''
任务描述：这是一个5分类任务，第1个数最大，类别为0
                       第2个数最大，类别为1
                       第3个数最大，类别为2
                       第4个数最大，类别为3
                       第5个数最大，类别为4
'''
epoch_number = 500
batch_size = 20
learn_rating = 0.001
train_sample = 5000
input_size = 5
out_size = 5


class TorchModel(nn.Module):
    def __init__(self, input_size,output_size):
        super(TorchModel, self).__init__()
        self.linear1 = nn.Linear(input_size,output_size)
        self.activation1 = torch.sigmoid
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.linear1(x)
        y_pre = self.activation1(x)

        if y is not None:
            return self.loss(y_pre, y)
        else:
            return y_pre


def set_data():
    x = np.random.random(5)
    x = x.tolist()
    if x.index(max(x)) == 0:
        return x, 0
    elif x.index(max(x)) == 1:
        return x, 1
    elif x.index(max(x)) == 2:
        return x, 2
    elif x.index(max(x)) == 3:
        return x, 3
    else:
        return x, 4


def total_data(number):
    X = []
    Y = []
    for i in range(number):
        x, y = set_data()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))


def evaluate(model):
    test_sample = 100
    correct, wrong = 0, 0
    x, y = total_data(test_sample)
    model.eval()
    with torch.no_grad():
        y_pre = model(x)
        y = y.numpy().tolist()
        y_pre = y_pre.numpy().tolist()
        for y_p, y_t in zip(y_pre, y):
            if y_p.index(max(y_p)) == y_t:
                correct += 1
            else:
                wrong += 1

    return correct / (correct + wrong)


def main():
    X, Y = total_data(train_sample)
    model = TorchModel(input_size,out_size)
    optim = torch.optim.Adam(model.parameters(), lr=learn_rating)

    epoch_loss = []
    acc_save = []
    for i in range(epoch_number):
        watch_loss = []
        model.train()
        for batch in range(train_sample // batch_size):
            x = X[batch_size * batch: batch_size * (batch + 1)]
            y = Y[batch_size * batch: batch_size * (batch + 1)]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())

        epoch_loss.append(np.mean(watch_loss))
        acc = evaluate(model)
        acc_save.append(float(acc))
        print("\n第%d轮的损失为：%f,正确率为：%f" % (i + 1, np.mean(np.array(watch_loss)),float(acc)))
    torch.save(model.state_dict(), "model.pt")

    plt.plot(epoch_loss, label="loss")
    plt.plot(acc_save, label="accurary")
    plt.legend()
    plt.show()


def verifacation(model_path, input_size, out_szie, test_data):
    model = TorchModel(input_size,out_szie)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(test_data))
        result = result.numpy().tolist()
        test_data = test_data.tolist()
    for data,result_pre in zip(test_data,result):
        print("输入数据为：%s，真实类别为：%d，预测类比为:%d" % (data,data.index(max(data)),result_pre.index(max(result_pre))))


if __name__ == "__main__":
    main()

    # test_data = np.random.random(25).reshape(5,5)
    # verifacation("model.pt", input_size,out_size,test_data)

