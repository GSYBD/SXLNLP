import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(0)
class_1 = np.random.rand(300, 5) * 0.5 + 0.5
class_2 = np.random.rand(300, 5) * 0.5 + 1.5
class_3 = np.random.rand(300, 5) * 0.5 + 2.5
dataset = np.vstack([class_1, class_2, class_3])
labels = np.array([0] * 300 + [1] * 300 + [2] * 300)

# print(dataset)
# print(labels)
indices = np.random.permutation(len(dataset))
# print(indices)
split_index = int(0.8 * len(dataset))
train_data, train_labels = dataset[:split_index], labels[:split_index]
val_data, val_labels = dataset[split_index:], labels[split_index:]
# 训练集
train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.long))
print(train_dataset)
# 验证集
val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.long))
# 训练集loader
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
# 验证集loader
val_loader = DataLoader(val_dataset, batch_size=10)
class Model(nn.Module):
    def __init__(self, input_size, hidden_size1):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y)
        else:
            return x


def evaluate(model):
    model.eval()
    for x, y in val_loader:
        y_pred = model(x)
        # print(y_pred, 'y_pred')
        y_pred_cls = torch.max(y_pred, 1)[1]
        # print(y_pred_cls, 'y_pred_cls')
        # print(y, 'y')
        # print(y_pred_cls == y, 'y_pred_cls == y')
        acc = (y_pred_cls == y).sum().item() / len(y)
        print("acc:%f" % acc)


def main():
    epoch_num = 200
    input_size = 5
    hidden_size1 = 3
    learning_rate = 0.001
    model = Model(input_size, hidden_size1)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    watch_loss = []
    for epoch in range(epoch_num):
        model.train()
        for x, y in train_loader:
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        evaluate(model)


if __name__ == "__main__":
    main()
