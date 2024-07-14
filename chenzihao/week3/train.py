from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from torch.optim import Adam
import torch

import preprocess
from model import Model

train_dataset, val_dataset, test_dataset \
    = preprocess.get_dataset("cnews/cnews.train.txt", \
                             "cnews/cnews.val.txt", \
                             "cnews/cnews.test.txt", \
                             "vocab.txt")
vocab = preprocess.read_vocab("vocab.txt")

indices = list(range(1000))
# subset = Subset(dataset, indices)
train_dataset = Subset(train_dataset,indices)
indices = list(range(200))
val_dataset = Subset(val_dataset,indices)
# test_dataset = test_dataset[:200]


# 如果batch_size非1还需要padding
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
# 初始化模型、损失函数和优化器
model = Model(len(vocab), 128, 256, 10)
criterion = nn.CrossEntropyLoss()

optimizer = Adam(model.parameters(),lr = 0.001)

# 训练函数
def train(epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

# 评估函数
def evaluate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Validation Loss: {running_loss / len(val_loader)}, Accuracy: {100 * correct / total}%')


print("*********************")
print(len(vocab))
print("*********************")
train(3)
evaluate()

# output is 

# Epoch 1, Loss: 0.016866044546247395
# Epoch 2, Loss: 0.030324721131406024
# Epoch 3, Loss: 0.08899538990798578
# Validation Loss: 0.038041401018138, Accuracy: 99.5%

# however, there is a segmentation fault after y ?
