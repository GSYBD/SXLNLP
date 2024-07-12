import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt


"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第x中那个维度大就输出那个下标
"""


class MultiClassModel(nn.Module):
    def __init__(self, input_size):
        super(MultiClassModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def build_sample():
    x = np.random.random(5)
    return x, int(np.argmax(x))


def build_dataset(num_samples):
    X = []
    Y = []
    for i in range(num_samples):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


def evaluate(model):
    model.eval()
    evaluate_dataset_num = 1000
    evaluate_x, evaluate_y = build_dataset(evaluate_dataset_num)
    print(f"本次评估集中各类别数量如下：\n0: {(evaluate_y == 0).sum()}\n1: {(evaluate_y == 1).sum()}\n2: {(evaluate_y == 2).sum()}\n3: {(evaluate_y == 3).sum()}\n4: {(evaluate_y == 4).sum()}")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(evaluate_x)
        print(y_pred)
        for y_p, y_t in zip(y_pred, evaluate_y):
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print(f"本次评估数据集中，正确数量：{correct}, 错误数量：{wrong}")
    return correct / (correct + wrong)


def train():
    epochs = 200
    train_dataset_num = 5000
    input_size = 5
    learning_rate = 0.001
    batch_size = 20

    model = MultiClassModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset(train_dataset_num)
    print(train_x, train_y)

    for epoch in range(epochs):
        model.train()
        watch_loss = []
        for batch_index in range(train_dataset_num // batch_size):
            batch_train_x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            batch_train_y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model.forward(batch_train_x, batch_train_y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print(f"===========================\n第{epoch + 1}轮的平均损失：{np.mean(watch_loss)}")
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    torch.save(model.state_dict(), "multi_class_model.pt")
    print(log)
    plt.plot(range(len(log)), [i[0] for i in log], label="acc")
    plt.plot(range(len(log)), [i[1] for i in log], label="loss")
    plt.legend()
    plt.show()


def predict(model_path, input_vec):
    input_size = 5
    model = MultiClassModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        # result = model.forward(torch.FloatTensor(input_vec))
        print(input_vec)
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        print(f"输入向量：{vec}, 预测类别：{torch.argmax(res)}, 预测概率：{res}")


if __name__ == '__main__':
    train()
    test_input_vec = build_dataset(5)
    # print(test_input_vec)
    predict("multi_class_model.pt", test_input_vec[0])
