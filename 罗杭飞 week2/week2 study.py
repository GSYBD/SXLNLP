import torch
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
X = []
Y = []

for _ in range(2000):
    x = np.random.randint(1, 11, size=5)
    max_indices = np.where(x == np.max(x))[0]
    y = np.zeros(len(x), dtype=int)
    y[max_indices] = 1
    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)
X_tensor = torch.from_numpy(X).float()
Y_tensor = torch.from_numpy(Y).float()

class TorchModel(torch.nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, 10)
        self.layer2 = torch.nn.Linear(10, 5)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

def validate_model(model, test_vec):
    model.eval()
    correct = 0
    total = len(test_vec)

    for x in test_vec:
        max_indices = np.where(np.array(x) == np.max(np.array(x)))[0]
        y = np.zeros(len(x), dtype=int)
        y[max_indices] = 1

        x_tensor = torch.from_numpy(np.array(x)).float().unsqueeze(0)
        y_tensor = torch.from_numpy(np.array(y)).float().unsqueeze(0)

        with torch.no_grad():
            output = model(x_tensor)
            predicted = (torch.sigmoid(output) > 0.5).int()

        print(f'输 入: {x}')
        print(f'预测值: {predicted.squeeze().numpy()}')
        print(f'真实值: {y}')
        print('---')

        if torch.equal(predicted, y_tensor.int()):
            correct += 1

    accuracy = correct / total
    print(f'准确率: {accuracy * 100:.2f}%')

def main():
    epoch_num = 20
    batch_size = 20
    learning_rate = 0.01
    input_size = 5
    model = TorchModel(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.BCEWithLogitsLoss()
    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []

    for epoch in range(epoch_num):
        model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch} ,Batch {batch_idx}, Loss {loss.item()}')

    torch.save(model.state_dict(), "model.bin")

    # 绘制loss变化图
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss During Training')
    plt.legend()
    plt.show()

    # 验证模型
    test_vec = [
        [3, 1, 4, 1, 5],
        [10, 3, 7, 2, 5],
        [4, 6, 8, 10, 3],
        [1, 7, 10, 5, 2],
        [9, 4, 3, 6, 8],
        [2, 10, 7, 4, 1],
        [5, 3, 8, 10, 6],
        [7, 2, 9, 4, 1],
        [6, 10, 3, 8, 5],
        [4, 1, 7, 2, 9],
        [8, 5, 10, 6, 3],
        [9, 7, 2, 1, 4],
        [3, 6, 10, 5, 8],
        [4, 2, 7, 9, 1],
        [5, 8, 1, 10, 6],
        [7, 3, 4, 2, 9],
        [6, 10, 5, 8, 1],
        [9, 4, 7, 3, 2],
        [1, 8, 5, 6, 10],
        [2, 7, 4, 9, 3]]
    validate_model(model, test_vec)

if __name__ == "__main__":
    main()
