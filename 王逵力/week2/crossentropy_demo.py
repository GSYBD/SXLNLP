import torch
import torch.nn as nn
import torch.optim as optim

def train_and_evaluate(x, y):
    class SimpleModel(nn.Module):
        def __init__(self, input_units, output_units):
            super(SimpleModel, self).__init__()
            self.linear = nn.Linear(input_units, output_units)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel(input_units=1, output_units=2)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_function(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    model.eval()
    predictions = model(x)
    predicted_labels = predictions.argmax(dim=1)
    accuracy = (predicted_labels == y).float().mean()
    print(f'Accuracy: {accuracy * 100:.2f}%')

# 手动定义输入数据x和标签y
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([0, 1, 0, 1, 0])

# 调用函数进行训练和评估
train_and_evaluate(x, y)