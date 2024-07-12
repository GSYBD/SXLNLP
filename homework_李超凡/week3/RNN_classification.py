import numpy as np
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# 构造数据集
def construct_dataset(dataset_path):
    words_ls = []
    classification_ls = []
    label_ls = []
    max_words_len = 0
    with open(dataset_path, "r", encoding="utf-8") as file:
        for line in file:
            line_split = line.split()
            if len(line_split) > 2:
                label_ls.append(int(line_split[0]))
                classification_ls.append(line_split[1])
                words = "".join(line_split[2:])
                words_ls.append(words)
                if len(words) > max_words_len:
                    max_words_len = len(words)
    label_classification_dict = get_label_classification_dict(label_ls, classification_ls)
    vocabulary_dict = get_vocabulary(words_ls)
    x_arr = np.zeros((len(words_ls), max_words_len))
    for i in range(len(words_ls)):
        x_arr[i, -1 * len(words_ls[i]):] = np.array([vocabulary_dict[character] for character in words_ls[i]])
    return x_arr, np.array(label_ls), label_classification_dict, vocabulary_dict


def get_label_classification_dict(label_ls, classification_ls):
    # 获取label和类型标签
    label_classification_dict = {}
    for i in range(len(label_ls)):
        if label_ls[i] not in label_classification_dict.keys():
            label_classification_dict[label_ls[i]] = classification_ls[i]
    return label_classification_dict


def get_vocabulary(words_ls):
    # 获取词表
    vocabulary_dict = {"pad": 0}
    characters_ls = sorted(list(set("".join(words_ls))))
    for index, character in enumerate(characters_ls):
        vocabulary_dict[character] = index + 1
    vocabulary_dict["unk"] = len(vocabulary_dict)
    return vocabulary_dict

# 搭建模型
class TorchRNN(nn.Module):
    def __init__(self, vector_dim, vocab, num_classes):
        super(TorchRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  # (vocab_len,vector_dim)
        self.rnn = nn.RNN(vector_dim, 2 * vector_dim, 1, bias=False,
                          batch_first=True)  # (input_size, hidden_size, num_layers)
        self.fc = nn.Linear(2 * vector_dim, num_classes)  # (hidden_size,num_classes)
        self.loss = nn.functional.cross_entropy  # 交叉熵自带softmax

    def forward(self, x, y=None):
        x = self.embedding(x)
        x, hidden = self.rnn(x)
        y_pred = self.fc(x[:, -1, :])
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

# 拆分训练集和测试集
def split_train_test(X, Y, test_ratio):
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=test_ratio)
    return torch.LongTensor(train_x), torch.LongTensor(train_y), torch.LongTensor(test_x), torch.LongTensor(test_y)


# 训练模型
def train_model():
    # 数据集构造
    dataset_path = "dataset/cnews_title/Train.txt"
    X, Y, label_classification_dict, vocabulary_dict = construct_dataset(dataset_path)

    # 参数定义
    test_ratio = 0.2
    epoch_num = 20
    batch_size = 64
    vector_dim = 20
    num_classes = 14
    learning_rate = 0.01

    # 数据准备
    train_x, train_y, test_x, test_y = split_train_test(X, Y, test_ratio)
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 模型与优化器实例化
    model = TorchRNN(vector_dim, vocabulary_dict, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 训练过程
    for epoch in tqdm(range(epoch_num), desc="epoch number"):
        model.train()
        watch_loss = []
        for batch_x, batch_y in train_loader:
            optim.zero_grad()  # 梯度归零
            loss = model(batch_x, batch_y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        accuracy = evaluate(model, test_loader)
        log.append([accuracy, np.round(float(np.mean(watch_loss)), 5)])

    # 训练过程可视化
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocabulary_dict, ensure_ascii=False, indent=2))
    writer.close()
    return

# 测试集评估
def evaluate(model, test_loader):
    model.eval()
    correct, wrong = 0, 0
    with torch.no_grad():  # 不计算梯度，降低内存和计算量开支
        for batch_x, batch_y in test_loader:
            y_predict = model(batch_x)
            for y_p, y_t in zip(y_predict, batch_y):
                if torch.argmax(y_p) == int(y_t):
                    correct += 1
                else:
                    wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


if __name__ == "__main__":
    train_model()
