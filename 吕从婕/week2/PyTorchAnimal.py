import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

num_samples = 100
input_size = 3 * 64 * 64
num_classes = 3  # 猫、狗、猪

X = torch.randn(num_samples, input_size)
y = torch.randint(0, num_classes, (num_samples,))

dataset = TensorDataset(X, y)

batch_size = 10
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class SimpleCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleCNN, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

model = SimpleCNN(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 5
for epoch in range(num_epochs):
    for inputs, labels in data_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
