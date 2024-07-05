import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json

# 定义RNN模型
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # 取序列最后一个输出
        x = self.fc(x)
        return x

# 构建词表
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {char: i+1 for i, char in enumerate(chars)}
    vocab['unk'] = len(vocab) + 1
    return vocab

# 生成样本数据
def build_sample(vocab, sentence_length, target_char='a', force_include=False):
    if force_include:
        # 强制包含目标字符
        x = [target_char] + [random.choice(list(vocab.keys())) for _ in range(sentence_length - 1)]
        random.shuffle(x)  # 随机打乱顺序
    else:
        x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if target_char in x:
        y = x.index(target_char)
    else:
        y = sentence_length
    x = [vocab.get(char, vocab['unk']) for char in x]
    return x, y

# 构建数据集
def build_dataset(sample_size, vocab, sentence_length, target_char='a', force_include=False):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_size):
        x, y = build_sample(vocab, sentence_length, target_char, force_include)
        dataset_x.append(x)
        dataset_y.append(y)
    dataset_x = torch.LongTensor(dataset_x)
    dataset_y = torch.LongTensor(dataset_y)
    return dataset_x, dataset_y

# 训练模型
def train(model, optimizer, criterion, dataset_x, dataset_y, epochs=10, batch_size=20):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for start in range(0, len(dataset_x), batch_size):
            end = start + batch_size
            x = dataset_x[start:end]
            y = dataset_y[start:end]
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / (len(dataset_x) / batch_size)}')

# 验证模型
def evaluate(model, dataset_x, dataset_y, vocab):
    model.eval()
    with torch.no_grad():
        outputs = model(dataset_x)
        probabilities = nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)
        for i in range(len(dataset_x)):
            input_indices = dataset_x[i].tolist()
            sentence = ''.join([list(vocab.keys())[list(vocab.values()).index(index)] for index in input_indices if index in vocab.values()])
            print(f"Sentence: '{sentence}', Predicted Position: {predicted[i].item()}, Actual Position: {dataset_y[i].item()}, Probability: {probabilities[i][predicted[i]].item()}")

def main():
    char_dim = 20  # Embedding大小
    sentence_length = 6  # 字符串长度
    hidden_size = 64  # RNN隐藏层大小
    output_size = sentence_length + 1  # 输出的类别数
    sample_size = 2000  # 样本数量
    validate_size = 5  # 验证数据集大小
    target_char = 'a'  # 我们要检测的特定字符
    vocab = build_vocab()  # 创建词汇表
    
    dataset_x, dataset_y = build_dataset(sample_size, vocab, sentence_length, target_char)
    validate_x, validate_y = build_dataset(validate_size, vocab, sentence_length, target_char, force_include=True)

    model = RNNModel(len(vocab) + 2, char_dim, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    train(model, optimizer, criterion, dataset_x, dataset_y, epochs=10, batch_size=20)
    evaluate(model, validate_x, validate_y, vocab)

    # 保存模型和词表
    torch.save(model.state_dict(), "rnn_model.pth")
    with open("vocab.json", "w") as f:
        json.dump(vocab, f)

if __name__ == "__main__":
    main()
