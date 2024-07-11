import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""

曹哲鸣第二周作业——对输入向量进行分类

规律：x是一个5维向量，判断向量中哪个元素的值最大，第几位元素值最大则该向量为第几类
例如：[4,6,9,1,5]，该向量中第三位的元素值最大，则该向量为第三类

"""

#选择模型
class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.layer = nn.Linear(input_size, 5)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        y_pred = self.layer(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

#生成随机样本数据
def CreateSample():
    x = np.random.random(5)
    y = np.zeros(x.shape)

    for i in range(len(x)):
        if x[i] == np.max(x):
            y[i] = 1
    return x, y

#样本集
def SampleDataSet(SampleSize):
    X = []
    Y = []
    for i in range(SampleSize):
        x, y = CreateSample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)

#测试模型
def evaluate(model):
    test_sample = 100
    correct, wrong = 0, 0
    x, y = SampleDataSet(test_sample)
    model.eval()
    with torch.no_grad():
        result = model(x)
        for y_pred, y in zip(result, y):
            if np.argmax(y_pred) == np.argmax(y):
                correct += 1
            else:
                wrong += 1
        print("测试样本数共有：%d，测试正确数量为：%d，正确率为%f" %(test_sample, correct, correct/(correct+wrong)))
        return correct/(correct+wrong)



#训练过程
def main():
    epoch_size = 50
    batch_size = 20
    sample_size = 5000
    input_size = 5
    lr = 0.001

    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    x_tarin, y_tarin = SampleDataSet(sample_size)
    log = []

    for epoch in range(epoch_size):
        model.train()
        watch_loss = []
        for batch_index in range(sample_size // batch_size):
            x = x_tarin[batch_index*batch_size : (batch_index + 1)*batch_size]
            y = y_tarin[batch_index*batch_size : (batch_index + 1)*batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())

        print("第%d轮训练的loss均值为%f" %(epoch+1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, np.mean(watch_loss)])

    torch.save(model.state_dict(), "homework.pt")
    plt.plot(range(len(log)), [l[0] for l in log], label = "acc")
    plt.plot(range(len(log)), [l[1] for l in log], label = "loss")
    plt.legend()
    plt.show()

#使用训练好的模型进行预测
def Predict(model_path, input):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result = model(torch.FloatTensor(input))
        for vec, y_pred in zip(input, result):
            print("输入的样本为：%s，预测结果为：%d类" %(vec, np.argmax(y_pred)+1))

if __name__ == "__main__":
    main()

    # input = [[0.23252395,0.18731718,0.08359847,0.76091346,0.97651414],
    #          [0.22065483,0.18586031,0.26641434,0.52817164,0.17684877],
    #          [0.39640023,0.22344796,0.62682448,0.79461963,0.59504811],
    #          [0.01256398,0.46449522,0.28803085,0.38516019,0.67485471],
    #          [0.25847699,0.86981325,0.39786142,0.68205988,0.74677569],
    #          [0.41959186,0.57710741,0.26487395,0.65685474,0.32465475]]
    #
    # Predict("homework.pt", input)
