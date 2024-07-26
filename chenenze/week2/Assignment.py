import torch
import torch.nn as nn
import numpy as np
import random

class Classification_Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Classification_Model, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = torch.softmax
        self.loss = nn.functional.cross_entropy
    
    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x, dim=1)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_sample():
    X = np.random.random(5)
    Y = [x == np.max(X) for x in X]
    return X, Y

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)

def evaluate(model):
    model.eval()
    N = 1000
    X, Y = build_dataset(N)
    with torch.no_grad():
        Y_pred = model(X)
        acc = (Y_pred.argmax(dim=1) == Y.argmax(dim=1)).float().mean().item()
        print(f"Accuracy: {acc}")
    return acc

def main():
    model = Classification_Model(5, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epoch = 500
    samples = 5000
    bach_size = 100
    
    for i in range(num_epoch):
        model.train()
        X, Y = build_dataset(samples)
        train_loss = 0
        for j in range(0, samples, bach_size):
            X_batch = X[j:j + bach_size]
            Y_batch = Y[j:j + bach_size]
            loss = model(X_batch, Y_batch)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item() * bach_size / samples
        print(f"Epoch {i}, Loss: {train_loss}")
        acc = evaluate(model)
        print(f"Accuracy: {acc}")
        if acc > 0.99: 
            break
        
    torch.save(model.state_dict(), "model.pth")
    return

def predict(model_path, input_vec):
    input_size = 5
    model = Classification_Model(input_size, input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(torch.argmax(res))), torch.max(res)))  # 打印结果
        
if __name__ == "__main__":
    main()
    
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    
    predict("model.pth", test_vec)
    