import random
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  # 嵌入层
        # 可以选择使用RNN或池化层
        # self.pool = nn.AvgPool1d(sentence_length)   # 池化层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # 循环神经网络层

        # 由于可能存在b不存在的情况，真实label设为sentence_length
        self.classify = nn.Linear(vector_dim, sentence_length + 1)  # 分类层
        self.loss = nn.functional.cross_entropy  # 损失函数

    def forward(self, x, y=None):
        x = self.embedding(x)
        # 使用RNN的情况
        rnn_out, hidden = self.rnn(x)
        x = rnn_out[:, -1, :]  # 取RNN最后一个时间步的输出

        y_pred = self.classify(x)  # 分类预测
        if y is not None:
            return self.loss(y_pred, y)  # 计算损失
        else:
            return y_pred  # 返回预测结果

def build_vocab():
    chars = "abcdefghijk"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)  # 未知字符的标号
    return vocab

def build_sample(vocab, sentence_length):
    x = random.sample(list(vocab.keys()), sentence_length)  # 随机不放回采样
    if "b" in x:  # 修改此处，寻找字符b的位置
        y = x.index("b")  # 修改此处
    else:
        y = sentence_length  # 如果没有b，则标签为句子长度
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字符转换为序号
    return x, y

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    epoch_num = 20
    batch_size = 40
    train_sample = 1000
    char_dim = 30
    sentence_length = 10
    learning_rate = 0.001
    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for _ in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])

    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), "model.pth")
    with open("vocab.json", "w", encoding="utf8") as writer:
        json.dump(vocab, writer, ensure_ascii=False, indent=2)

def predict(model_path, vocab_path, input_strings):
    char_dim = 30
    sentence_length = 10
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    x = [[vocab[char] for char in input_string] for input_string in input_strings]
    model.eval()
    with torch.no_grad():
        result = model(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%s" % (input_string, torch.argmax(result[i]), result[i]))

if __name__ == "__main__":
    main()
    test_strings = ["kijbcdefh", "gijkbcdea", "gkijbadfec", "kijhdbeca"]
    predict("model.pth", "vocab.json", test_strings)