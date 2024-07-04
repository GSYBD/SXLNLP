import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class MultiClassficationModel(nn.Module):
    def __init__(self, input_size):
        super(MultiClassficationModel, self).__init__()
        # 4维输出，对应四个类别
        self.linear = nn.Linear(input_size, 4)
        # 使用nn.CrossEntropyLoss自动应用log_softmax和计算交叉熵
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # 直接输出logits
        y_pred = self.linear(x)
        if y is not None:
            # 计算损失时，y需要是类别索引的形式
            return self.loss(y_pred, y)
        else:
            # 返回模型预测的logits
            return y_pred

# 生成四维向量样本
def build_sample():
    x = np.random.random(4)
    max_index = np.argmax(x)
    return x, max_index

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model, test_x, test_y):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        predictions = model(test_x)
        predictions = torch.argmax(predictions, dim=1)
        correct = (predictions == test_y).sum().item()
        total = predictions.size(0)
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def main():
    epoch_num = 20
    batch_size = 64
    train_sample = 1000
    input_size = 4  # 输入向量维度是4
    learning_rate = 0.001

    model = MultiClassficationModel(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_x, train_y = build_dataset(train_sample)
    test_x, test_y = build_dataset(100)  # 生成一些测试数据

    for epoch in range(epoch_num):
        model.train()
        for i in range(0, train_x.size(0), batch_size):
            batch_x = train_x[i:i+batch_size]
            batch_y = train_y[i:i+batch_size]
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = model.loss(output, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epoch_num} Loss: {loss.item()}")

    evaluate(model, test_x, test_y)
    torch.save(model.state_dict(), "model.pt")

    return model

if __name__ == "__main__":
    model = main()
    test_vec = [
        [0.57889086, 0.35229675, 0.31082123, 0.023504317],
        [0.95963533, 0.51524256, 0.95758807, 0.95520434],
        [0.68797868, 0.67482528, 0.13625847, 0.34675372],
        [0.89349776, 0.59416669, 0.92579291, 0.41567412]
    ]
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    with torch.no_grad():
        predictions = model(torch.FloatTensor(test_vec))
        for vec, pred in zip(test_vec, predictions):
            pred_index = torch.argmax(pred).item()
            print(f"Input: {vec}, Predicted Index: {pred_index}, Probability: {pred.tolist()}")

