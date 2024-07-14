import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt

def sine_sample_1(): # sin(2x * pi)
    return [torch.sin(2*torch.pi*x) + np.random.standard_normal() for x in torch.linspace(0, 1, 100)]

def sine_sample_2(): # sin(x * pi)
    return [torch.sin(torch.pi*x) + np.random.standard_normal() for x in torch.linspace(0, 1, 100)]

def sine_sample_3(): # 2sin(5x * pi)
    return [2 * torch.sin(5*torch.pi*x) + np.random.standard_normal() for x in torch.linspace(0, 1, 100)]

def generate_samples():
    x1 = torch.tensor([sine_sample_1() for _ in range(200)])
    x2 = torch.tensor([sine_sample_2() for _ in range(200)])
    x3 = torch.tensor([sine_sample_3() for _ in range(200)])
    
    y1 = torch.full((200, 1), 0)
    y2 = torch.full((200, 1), 1)
    y3 = torch.full((200, 1), 2) 

    X = torch.vstack([x1, x2, x3])
    y = torch.vstack([y1, y2, y3]).view(-1)

    return X, y

class DataSet(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.nn.functional.one_hot(y, 3).float()
        self.len = self.X.shape[0]
  
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return self.len

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Model, self).__init__()

        self.rnn = torch.nn.RNN(input_size,hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, data, hidden=None):
        output, hn = self.rnn(data, hidden)

        return self.linear(output), hn
    
def evaluate(model, testloader):
    model.eval()
    correct, wrong = 0, 0
    hiden = None
    with torch.no_grad():
        for x_test, y_test in testloader:
            y_pred, hiden = model(x_test, hiden)  # 模型预测
            y_pred = torch.argmax(y_pred, dim=1)
            y_test = torch.argmax(y_test, dim=1)
            correct += (y_pred == y_test).sum().item()
            wrong += (y_pred != y_test).sum().item()
    print("测试集准确率：%f" % (correct / (correct + wrong)))
    return correct / (correct + wrong)


def train():
    X, y = generate_samples()
    dataset = DataSet(X, y)
    train_set, test_set = data.random_split(dataset, [500, 100])
    train_loader = data.DataLoader(train_set, batch_size=100, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=100)

    epoch_num = 2
    lr = 0.0005
    input_size = X.shape[1]
    hiden_layer = 128
    num_layers = 1
    output_size =3

    model = Model(input_size, hiden_layer, num_layers, output_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.functional.cross_entropy

    log = []
    hiden = None
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        
        for x, y in train_loader:
            output, hiden = model(x, hiden) 
            loss = loss_fn(output, y) # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            hiden = hiden.detach()

            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, testloader=test_loader)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

    return
    

train()