import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from faker import Faker
import matplotlib.pyplot as plt

# 构建词汇表，包括指定的关键字
def build_vocab():
    i_keywords = ["安静", "独处", "思考", "内向"]
    e_keywords = ["社交", "热闹", "活动"]
    all_keywords = i_keywords + e_keywords
    fake = Faker("zh_CN")
    fake_sentences = [fake.sentence(nb_words=10) for _ in range(1000)]
    char_counter = Counter(''.join(fake_sentences + all_keywords))
    vocab = {"pad": 0}
    for index, char in enumerate(char_counter.keys()):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab) + 1
    return vocab, i_keywords, e_keywords

# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, file_path, vocab):
        self.data = []
        self.labels = []
        self.vocab = vocab
        self.max_len = 0
        with open(file_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split('真实类别: ')
                if len(parts) == 2:
                    sentence = parts[0].replace('样本: ', '').strip()
                    label = 1 if 'I人' in parts[1] else 0
                    encoded_sentence = self.encode_sentence(sentence)
                    self.data.append(encoded_sentence)
                    self.labels.append(label)
                    if len(encoded_sentence) > self.max_len:
                        self.max_len = len(encoded_sentence)
                    print(f"样本: {sentence}, 真实类别: {'I人' if label == 1 else 'E人'}")
                    print(f"分词: {encoded_sentence}\n")
        print(f"数据集加载完成，包含 {len(self.data)} 条样本。")

    def encode_sentence(self, sentence):
        words = list(sentence)
        return [self.vocab.get(word, self.vocab['unk']) for word in words]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        padded_sentence = self.pad_sentence(self.data[idx])
        return torch.tensor(padded_sentence, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)
    
    def pad_sentence(self, sentence):
        return sentence + [self.vocab['pad']] * (self.max_len - len(sentence))

# 改进后的RNN模型
class ImprovedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size, embedding_dim, num_layers=2):
        super(ImprovedRNN, self).__init__()
        # 嵌入层：将单个字符的索引映射到低维度的向量空间
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        # RNN层：用来捕捉序列中的上下文信息
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)
        # 全连接层：将RNN的输出映射到分类结果
        self.fc = nn.Linear(hidden_size, output_size)
        # Dropout层：防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.dropout(x[:, -1, :])
        x = self.fc(x)
        return x

# 设置模型和训练参数
input_size = 100
hidden_size = 128
output_size = 2
embedding_dim = 128
num_layers = 2
num_epochs = 100
batch_size = 32
learning_rate = 0.001

# 生成数据
vocab, i_keywords, e_keywords = build_vocab()
dataset = TextDataset('samples.txt', vocab)
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
print(f"数据集总大小: {dataset_size}")
print(f"训练集大小: {train_size}")
print(f"测试集大小: {test_size}")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedRNN(input_size, hidden_size, output_size, len(vocab), embedding_dim, num_layers).to(device)
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# Adam优化器，用于自动调整学习率并优化模型参数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
train_accuracies = []
test_accuracies = []

print("开始训练...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / train_size
    train_accuracy = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(train_accuracy)
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # 计算测试集准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct / total
    test_accuracies.append(test_accuracy)

print(f"测试集上的准确率: {test_accuracy:.2f}%")

# 绘制损失和准确率图
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. Epoch')
plt.legend()

plt.tight_layout()
plt.show()

# 用户交互
def predict_sentence(sentence, model, vocab, device):
    model.eval()
    encoded_sentence = [vocab.get(char, vocab['unk']) for char in list(sentence)]
    padded_sentence = encoded_sentence + [vocab['pad']] * (dataset.max_len - len(encoded_sentence))
    input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)
        predicted_label = predicted.item()
    
    return 'I人' if predicted_label == 1 else 'E人'

print("输入一句话来预测其类别 (输入'q'退出)：")
while True:
    user_input = input("请输入一句话: ")
    if user_input.lower() == 'q':
        break
    prediction = predict_sentence(user_input, model, vocab, device)
    print(f"预测类别: {prediction}")
