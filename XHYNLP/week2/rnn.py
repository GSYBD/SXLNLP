import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import random
# characters = ['你', '我', '他', '她', '它', '春', '夏', '秋', '冬', '江', '河', '湖', '海', '日', '月', '星', '辰', '东', '西', '南', '北', '风', '雨', '云', '雷', '电', '火', '水', '土', '石']
# # 生成包含“你”在不同位置的四字词数据集
# phrases = [
#     f'{c1}{c2}你{c3}' for c1, c2, c3 in zip(characters[1:], characters[1:], characters[1:])
# ] + [
#     f'{c1}你{c2}{c3}' for c1, c2, c3 in zip(characters[1:], characters[1:], characters[1:])
# ] + [
#     f'你{c1}{c2}{c3}' for c1, c2, c3 in zip(characters[1:], characters[1:], characters[1:])
# ] + [
#     f'{c1}{c2}{c3}你' for c1, c2, c3 in zip(characters[1:], characters[1:], characters[1:])
# ]
characters = ['你', '我', '他', '她', '它', '春', '夏', '秋', '冬', '江', '河', '湖', '海', '日', '月', '星', '辰', '东', '西', '南', '北', '风', '雨', '云', '雷', '电', '火', '水', '土', '石']
# 字符编码
char_to_idx = {char: idx for idx, char in enumerate(characters)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
# 去除“你”，以免在组合中重复使用
characters.remove('你')

# 创建一个空列表来存储生成的短语及其对应的标签
phrases_with_labels = []

# 遍历所有可能的位置插入“你”
for position in range(0, 5):
    # 对于每个位置，随机选择三个字符并生成短语
    for _ in range(len(characters)):
        other_chars = random.sample(characters, 3)
        phrase = ''
        if position == 1:
            phrase = '你' + ''.join(other_chars)
        elif position == 2:
            phrase = other_chars[0] + '你' + ''.join(other_chars[1:])
        elif position == 3:
            phrase = other_chars[0] + other_chars[1] + '你' + other_chars[2]
        elif position == 4:
            phrase = ''.join(other_chars) + '你'
        else :
            other_chars = random.sample(characters, 4)
            phrase = ''.join(other_chars) 
        phrases_with_labels.append((phrase, position))

# 打印生成的短语及其对应的标签
for phrase, label in phrases_with_labels:
    print(f'({phrase}, {label})')
data = phrases_with_labels
# 构建数据集和标签
# data = []
# for i, phrase in enumerate(phrases_with_labels):
#     if '你' in phrase:
#         data.append((phrase, phrases_with_labels.index(phrase) % 4 + 1))
#     else:
#         data.append((phrase, 0))
print(data)

# 数据转换为数字序列和标签
inputs = []
targets = []
for phrase, label in data:
    encoded_phrase = [char_to_idx[char] for char in phrase]
    inputs.append(encoded_phrase)
    targets.append(label)
# 转换为张量
inputs_tensor = torch.tensor(inputs).long()
targets_tensor = torch.tensor(targets).long()
# 创建数据加载器
dataset = TensorDataset(inputs_tensor, targets_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# 定义模型
class PositionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(PositionClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.rnn(embedded)
        out = self.fc(out[:, -1, :])  # 取序列最后一个时间点的输出
        return out
vocab_size = len(characters)+1
embedding_dim = 16
hidden_size = 32
output_size = 5  # 分类标签从0到4
model = PositionClassifier(vocab_size, embedding_dim, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        
torch.save(model.state_dict(), './model_weights.pth')
print("Model parameters saved to model_weights.pth")
model = PositionClassifier(vocab_size, embedding_dim, hidden_size, output_size)
model.load_state_dict(torch.load('./model_weights.pth'))      
# 测试模型
data_test = [
    ('你雷我风', 1),
    ('月你星雨', 2),
    ('辰东你云', 3),
    ('电水石你', 4),
    ('春夏秋冬', 0),  
    ('江河湖海', 0),  
    ('你我他她', 1),
    ('他你我她', 2),
    ('他她你我', 3),
    ('他她我你', 4)
]
inputs_test = []
targets_test = []
for phrase, label in data_test:
    encoded_phrase = [char_to_idx[char] for char in phrase]
    inputs_test.append(encoded_phrase)
    targets_test.append(label)
# 转换为张量
inputs_tensor = torch.tensor(inputs_test).long()
targets_tensor = torch.tensor(targets_test).long()
dataset = TensorDataset(inputs_tensor, targets_tensor)
dataloader = DataLoader(dataset, batch_size=len(inputs_test))
correct = 0
total = 0
y_pred = []
with torch.no_grad():
    for inputs,targets in dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        y_pred.extend(predicted)
print(f"Accuracy on test set: {100 * correct / total}%")    
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix   
cm = confusion_matrix(targets_test, y_pred)
classes = ['Class 0', 'Class 1','Class 2','Class 3','Class 4']
plt.figure(figsize=(12, 8))
# 绘制混淆矩阵
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(range(len(classes)), classes, rotation=45) 
plt.yticks(range(len(classes)), classes)
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('confusion_matrix.png')
print("Confusion matrix image saved to confusion_matrix.png")