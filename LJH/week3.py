import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
import string

"自己设计文本任务目标，使用rnn进行多分类"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.eb = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.rnn = nn.RNN(vector_dim, sentence_length, bias=True, batch_first=True)
        self.bn = nn.BatchNorm1d(sentence_length)
        self.fc = nn.Linear(sentence_length, 4)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.eb(x)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.bn(x)
        y_pred = self.fc(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def generate_string(length):
    letters = list(string.ascii_lowercase)
    random_string = ''.join(random.choice(letters) for _ in range(length))
    return random_string


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {'pad': 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if set('a') & set(x):
        y = 1
    elif set('i') & set(x):
        y = 2
    elif set('n') & set(x):
        y = 3
    else:
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(char_dim, sentence_length, vocab):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(500, vocab, sample_length)
    x, y = x.to(device), y.to(device)
    correct, wrong = 0, 0
    i = np.count_nonzero(y.tolist())
    print("本次预测集中共有%d个正样本，%d个负样本" % (i, 500 - i))
    with torch.no_grad():
        y_pred = model(x).to(device)
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == int(y_t):
                correct += 1  # 负样本判断正确
            else:
                wrong += 1
        print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
        return correct / (correct + wrong)


def main():
    sentence_length = 6  # 样本文本长度
    epoch_num = 30  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    learning_rate = 0.005  # 学习率
    char_dim = 15  # 每个字的维度
    vocab = build_vocab()  # 建立字表
    model = build_model(char_dim, sentence_length, vocab).to(device)  # 建立模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 选择优化器
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            x, y = x.to(device), y.to(device)
            optim.zero_grad()  # 梯度归零
            loss = model(x, y).to(device)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), "torchmodel.pth")  # 保存词表
    return


def predict(model_path, vocab_path, input_strings):
    char_dim = 15
    sentence_length = 6
    vocab = json.load((open(vocab_path, 'r', encoding='utf8')))
    model = build_model(char_dim, sentence_length, vocab)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    x = torch.LongTensor(x).to(device)
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(x)  # 模型预测
    for vec, res in zip(input_strings, result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res))  # 打印结果


if __name__ == "__main__":
    main()
    test_strings = []
    for i in range(20):
        x = generate_string(6)
        test_strings.append(x)
    predict("torchmodel.pth", "vocab.json", test_strings)
