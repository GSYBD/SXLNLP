import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Simple dataset
# Features (input) and labels (target)
data = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 3.0]])
labels = torch.tensor([0, 0, 1, 1])  # Binary labels

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(2, 2)  # Input size is 2, output size is 2 (for binary classification)

    def forward(self, x):
        return self.fc(x)

# Create the network
net = SimpleNN()

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()  # This is the cross entropy loss function
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Implement softmax function
def softmax(matrix):
    return np.exp(matrix) / np.sum(np.exp(matrix), axis=1, keepdims=True)

# Convert input to one-hot matrix
def to_one_hot(target, shape):
    one_hot_target = np.zeros(shape)
    for i, t in enumerate(target):
        one_hot_target[i][t] = 1
    return one_hot_target

# Manually implement cross entropy
def cross_entropy(pred, target):
    batch_size, class_num = pred.shape
    pred = softmax(pred)
    target = to_one_hot(target, pred.shape)
    entropy = - np.sum(target * np.log(pred), axis=1)
    return np.sum(entropy) / batch_size

# Train the network
for epoch in range(50):  # loop over the dataset 50 times
    optimizer.zero_grad()   # zero the parameter gradients
    outputs = net(data)     # forward pass
    loss = criterion(outputs, labels)  # compute cross entropy loss
    loss.backward()         # backward pass
    optimizer.step()        # optimize

    if epoch % 10 == 9:     # print every 10 epochs
        print(f'Epoch {epoch + 1}, Loss: {loss.item():.3f}')

        # Manual calculation of cross entropy loss
        outputs_np = outputs.detach().numpy()
        labels_np = labels.numpy()
        manual_loss = cross_entropy(outputs_np, labels_np)
        print(f'Epoch {epoch + 1}, Manual Loss: {manual_loss:.3f}')

print('Finished Training')
