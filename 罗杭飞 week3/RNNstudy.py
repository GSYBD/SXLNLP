import random
import string
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


# 生成随机单词
def generate_random_words(num_words, word_length):
    return [''.join(random.choices(string.ascii_lowercase, k=word_length)) for _ in range(num_words)]


# 步骤1：生成随机单词，每个单词由5个字母组成
words = generate_random_words(2000, 5)

# 步骤2：为单词贴标签（1表示正例，0表示负例）
labels = [1 if any(c in word for c in 'xyz') else 0 for word in words]

# 将单词和标签组合成一个元组列表
data = list(zip(words, labels))

# 创建一个DataFrame
df = pd.DataFrame(data, columns=['Word', 'Label'])

# 保存为CSV文件（将路径更改为你的指定目录）
output_path = "C:\\Users\\Administrator\\Desktop\\NLP学习\\week3 深度学习处理文本\\random_words_labels.csv"
df.to_csv(output_path, index=False)

# 步骤3：读取CSV文件
df_read = pd.read_csv(output_path)

# 步骤5：构建词汇表
vocab = {char: idx for idx, char in enumerate(string.ascii_lowercase)}

# 步骤6：定义一个将字符串转换为索引序列的函数
def str_to_sequence(string, vocab):
    return [vocab[char] for char in string]

# 测试字符串转换函数
test_string = "sdadfafdsf"
sequence = str_to_sequence(test_string, vocab)
print(sequence)


# 步骤7：定义一个自定义Dataset
class WordsDataset(Dataset):
    def __init__(self, df, vocab):
        self.words = df['Word'].values
        self.labels = df['Label'].values
        self.vocab = vocab

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        label = self.labels[idx]
        sequence = str_to_sequence(word, self.vocab)
        return torch.tensor(sequence), torch.tensor(label)


# 创建数据集和数据加载器
dataset = WordsDataset(df_read, vocab)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# 步骤4：初始化嵌入层
num_embeddings = 26  # 字母表中字符的数量
embedding_dim = 5  # 每个字符向量的维度
embedding_layer = nn.Embedding(num_embeddings, embedding_dim)


# 步骤8：定义RNN模型
class RNNModel(nn.Module):
    def __init__(self, embedding_layer):
        super(RNNModel, self).__init__()
        self.embedding = embedding_layer
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=10, batch_first=True)
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        x = self.embedding(x)
        rnn_out, _ = self.rnn(x)
        rnn_out = rnn_out[:, -1, :]
        out = self.fc(rnn_out)
        return out


# 初始化模型、损失函数和优化器
model = RNNModel(embedding_layer)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 记录损失值和准确率
loss_values = []
accuracy_values = []

# 步骤9：训练RNN模型
num_epochs = 15
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0

    for sequences, labels in dataloader:
        labels = labels.float().unsqueeze(1)
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = torch.round(torch.sigmoid(outputs))
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / len(dataloader)
    accuracy = correct / total

    loss_values.append(average_loss)
    accuracy_values.append(accuracy)

    print(f'第 {epoch + 1}/{num_epochs} 轮，损失值: {average_loss:.4f}，准确率: {accuracy:.4f}')


# 步骤10：可选 - 保存训练好的模型
model_path = "C:\\Users\\Administrator\\Desktop\\NLP学习\\week3 深度学习处理文本\\rnn_model.pth"
torch.save(model.state_dict(), model_path)


# 生成图表
epochs = range(1, num_epochs + 1)

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(epochs, loss_values, 'b-', label='Loss')
ax2.plot(epochs, accuracy_values, 'g-', label='Accuracy')

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='b')
ax2.set_ylabel('Accuracy', color='g')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.title('Loss and Accuracy over Epochs')
plt.show()


# 检测模型的正确率
def evaluate_model(model, words, labels, vocab):
    model.eval()
    correct = 0
    total = len(words)

    with torch.no_grad():
        for word, label in zip(words, labels):
            sequence = torch.tensor([str_to_sequence(word, vocab)])
            output = model(sequence)
            predicted = torch.round(torch.sigmoid(output))
            correct += (predicted.item() == label)

    accuracy = correct / total
    return accuracy


# 生成50个随机单词，检测模型正确率
test_words = generate_random_words(500, 5)
test_labels = [1 if any(c in word for c in 'xyz') else 0 for word in test_words]
accuracy = evaluate_model(model, test_words, test_labels, vocab)

print(f'模型在500个随机单词上的正确率: {accuracy:.4f}')
