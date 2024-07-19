import random

import torch
import torch.nn as nn
import torch.optim as optim

categories = ["chinese", "english", "number"]


def generate_random_string(category, length):
    if category == "chinese":
        return "".join(
            random.choice("八斗深度学习循环和卷积神经网络") for _ in range(length)
        )
    elif category == "english":
        return "".join(
            random.choice("badouAIdeeplearningrecurrentandconvotionalneuralnetwork")
            for _ in range(length)
        )
    elif category == "number":
        return "".join(random.choice("0123456789") for _ in range(length))


data = []
labels = []

for _ in range(10000):
    category = random.choice(categories)
    text_length = random.randint(1, 10)
    text = generate_random_string(category, text_length)
    data.append(text)
    labels.append(categories.index(category))

print(data[0:3], labels[0:3])

vocab = set()
for text in data:
    vocab.update(text)
vocab = list(vocab)
char_to_idx = {
    char: idx + 1 for idx, char in enumerate(vocab)
}  # 索引从1开始，0用于填充

data = [[char_to_idx[char] for char in text] for text in data]

max_length = max(len(text) for text in data)
data = [text + [0] * (max_length - len(text)) for text in data]

data = torch.tensor(data, dtype=torch.long)
labels = torch.tensor(labels, dtype=torch.long)

# 拆分训练集和测试集
train_data, test_data = data[:8000], data[8000:]
train_labels, test_labels = labels[:8000], labels[8000:]


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))


VOCAB_SIZE = len(vocab) + 1  # 包括填充值0
EMBEDDING_DIM = 100
HIDDEN_DIM = 200
OUTPUT_DIM = len(categories)

model = RNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = criterion.to(device)


def train(model, data, labels):
    model.train()
    optimizer.zero_grad()
    data, labels = data.to(device), labels.to(device)
    predictions = model(data)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


N_EPOCHS = 50
for epoch in range(N_EPOCHS):
    epoch_loss = train(model, train_data, train_labels)
    print(f"Epoch {epoch+1}/{N_EPOCHS}, Loss: {epoch_loss:.4f}")


def evaluate(model, data, labels):
    model.eval()
    data, labels = data.to(device), labels.to(device)
    with torch.no_grad():
        predictions = model(data)
        loss = criterion(predictions, labels)
    return loss.item()


test_loss = evaluate(model, test_data, test_labels)
print(f"Test Loss: {test_loss:.4f}")


def prepare_string(s):
    s_idx = [char_to_idx[char] for char in s]
    s_idx = s_idx + [0] * (max_length - len(s_idx))
    s_idx = torch.tensor(s_idx, dtype=torch.long).unsqueeze(0).to(device)
    return s_idx


def predict(s):
    model.eval()
    s_idx = [char_to_idx[char] for char in s]
    s_idx = s_idx + [0] * (max_length - len(s_idx))
    s_idx = torch.tensor(s_idx, dtype=torch.long).unsqueeze(0).to(device)
    output = model(s_idx)
    output = nn.Softmax(dim=1)(output)
    output = torch.argmax(output, dim=1).item()
    return output


test_string1 = "深度学习"
test_string2 = "deeplearning"
test_string3 = "123"

output1 = predict(test_string1)
output2 = predict(test_string2)
output3 = predict(test_string3)


print(f"Test string 1: {test_string1}, Predicted category: {categories[output1]}")
print(f"Test string 2: {test_string2}, Predicted category: {categories[output2]}")
print(f"Test string 3: {test_string3}, Predicted category: {categories[output3]}")
