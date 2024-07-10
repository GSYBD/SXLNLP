import torch
import torch.nn as nn
import numpy as np
"""
评估模型在测试集上的准确率。它通过计算预测输出与真实标签的比较，统计预测正确的数量，并计算准确率
"""
class SentimentAnalysisModel(nn.Module):             # 模型类
    def __init__(self, input_size, num_classes):
        super(SentimentAnalysisModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes) # 线性层
        self.loss = nn.CrossEntropyLoss()  # 交叉熵

    def forward(self, x, y=None):
        y_pred = self.linear(x)

        if y is not None:
            return self.loss(y_pred, y)  # 交叉熵
        else:
            return y_pred

def build_sample():          # 一组数据
    x = np.random.random(100)  # 100 个随机数
    label = np.random.randint(0, 3)  #  0, 1, 2
    return x, label

def build_dataset(total_sample_num):  # 多组数据
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()    # 创建一组随机数
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
def evaluate(model, test_sample_num):
    model.eval()
    x, y = build_dataset(test_sample_num)
    print(x)
    print(y)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        _, predicted = torch.max(y_pred.data, 1) # torch.max 输出张量最大值 torch.max()这个函数返回的是两个值，第一个值是具体的value（我们用下划线_表示，因为不关注，所以用_），第二个值是value所在的index（也就是predicted）
        correct += (predicted == y).sum().item() #  # 如果预测结果和真实值相等则计数 +1

    accuracy = correct / test_sample_num
    print("Accuracy: {:.2f}%".format(accuracy * 100)) # 准确率
    return accuracy

def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 100
    num_classes = 3
    learning_rate = 0.001

    model = SentimentAnalysisModel(input_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        epoch_loss = 0.0

        for batch_index in range(train_sample // batch_size):
            start = batch_index * batch_size
            end = (batch_index + 1) * batch_size
            x = train_x[start:end]
            y = train_y[start:end]

            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= (train_sample // batch_size)
        print("Epoch {}, Loss: {:.4f}".format(epoch+1, epoch_loss))
        accuracy = evaluate(model, test_sample_num=1000)
        log.append([accuracy, epoch_loss])

    torch.save(model.state_dict(), "model.pt")

if __name__ == "__main__":
    main()