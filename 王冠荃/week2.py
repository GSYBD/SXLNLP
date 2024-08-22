import torch
import torch.nn as nn
import numpy as np

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes=5):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        self.linear1 = nn.Linear(input_size, 10)
        self.linear2 = nn.Linear(10, num_classes)
        self.loss = nn.CrossEntropyLoss() 

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # x = self.linear1(x)  # (batch_size, input_size) -> (batch_size, 10)
        # x = torch.relu(x)
        # x = self.linear2(x)  # (batch_size, 10) -> (batch_size, num_classes)
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y.long().squeeze())  # 预测值和真实值计算损失
        else:
            return x  # 输出预测结果
        
def generate():
    x = np.random.random(5)
    y = np.argmax(x)
    return x, y

def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = generate()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

def test(model):
    model.eval()
    test_num = 100
    x, y = build_dataset(test_num)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)
        y_pred = torch.argmax(y_pred, dim=1)
        correct = (y_pred == y).sum().item()
    print("correct:", correct)
    return correct / test_num

def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率

    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_x, train_y = build_dataset(train_sample)
    batch_num = train_sample // batch_size
    for i in range(epoch_num):
        model.train()
        total_loss = 0
        for batch_index in range(batch_num):
            x = train_x[batch_size * batch_index:(batch_index + 1) * batch_size]
            y = train_y[batch_size * batch_index:(batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            total_loss += loss.item()
        print("%d epoch loss: %f" % (i + 1, total_loss/20))
        acc = test(model)
        print("acc:", acc)

    torch.save(model.state_dict(), "model.pt")
    print("model is saved.")

def predict(model_path, input):
    model = TorchModel(input_size=5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        res = model.forward(torch.FloatTensor(input))
        res = torch.argmax(res, dim=1)
    for x, r in zip(input, res):
        print("input %s, predict %d" % (x, r.item()))

if __name__ == "__main__":
    main()

    # x = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
    #     [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #     [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #     [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # model_path = "model.pth"
    # predict(model_path, x)
