import torch
import torch.nn as nn
import numpy as np
import random
import torch.utils.data as Data
import matplotlib.pyplot as plt
"""
目的：构造一个使用rnn/embed ing的多分类模型，对模型训练并打印
规律：随机生成含字符串，将满足交集的进行多分类
"""

class TorchModel(nn.Module):
    def __init__(self, hidden_size, vocab, sentence_length):
        super(TorchModel, self).__init__()
        self.emb = nn.Embedding(len(vocab)+1, hidden_size,padding_idx=0)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.pool = nn.MaxPool1d(sentence_length)
        self.classify = nn.Linear(hidden_size, 5)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.emb(x)
        x, h = self.rnn(x)  # h 已经是最后一个时间步的隐藏状态
        # 使用 h 而不是 x，因为我们使用的是最后一个隐藏状态
        y_pred = self.classify(h.squeeze(0))
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_vocab():
        chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
        vocab = {"pad": 0}
        for index, char in enumerate(chars):
            vocab[char] = index + 1  # 每个字对应一个序号
        #vocab['unk'] = len(vocab)  # 26
        return vocab

def build_sample(vocab, sentence_length):
        # 随机从字表选取sentence_length个字，可能重复
        x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]

        # 定义五组分类字符集
        groups = {
            1: set("abcde"),
            2: set("fghij"),
            3: set("klmno"),
            4: set("pqrst"),
            5: set("uvwxyz")
        }

        # 检查生成的字符串属于哪个分类
        for group_id, group_chars in groups.items():
            if group_chars & set(x):  # 如果x中有字符属于当前组
                y = group_id - 1  # 减1以适应0-based索引
                break
        else:  # 如果没有匹配任何一组，可以定义一个默认分类
            y = 0  # 假设0为默认分类

        # 将字转换成序号，为了做embedding
        x = [vocab.get(x[i]) for i in range(sentence_length)]
        return x, y

def build_dataset(train_simple, vocab, sentence_length):
    X = []
    Y = []
    for i in range(train_simple):
        x, y = build_sample(vocab, sentence_length)
        X.append(x)
        # 确保 Y 是一个一维列表
        Y.append(y)  # y 应该是一个整数，而不是列表

    return torch.LongTensor(X), torch.LongTensor(Y)  # Y 已经是一维的，直接转换

def evaluate(model, vocab, sentence_length):
    x, y = build_dataset(100, vocab, sentence_length)
    model.eval()
    correct, error = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_t, y_p in zip(y, y_pred):
            if int(y_t) == int(torch.argmax(y_p)):
                correct += 1
            else:
                error += 1
    print("正确预测个数：%d / %d, 正确率：%f" % (correct, correct + error, correct / (correct + error)))
    return correct / (correct + error)


def main():
    batch_size = 20
    lr = 0.002
    train_simple = 5000
    hidden_size = 100
    vocab = build_vocab()
    epoch_size = 10
    sentence_length = 4
    # build model
    model = TorchModel(sentence_length, vocab, hidden_size)
    # 優化器
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # 訓練的數據
    X, Y = build_dataset(train_simple, vocab, sentence_length)
    # 分割數據
    dataset = Data.TensorDataset(X, Y)
    data_item = Data.DataLoader(dataset, batch_size, shuffle=True)
    log = []
    for epoch in range(epoch_size):
        model.train()
        watch_loss = []
        model.train()
        for x, y_true in data_item:
            loss = model(x, y_true)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("第%d轮 loss = %f" % (epoch + 1, np.mean(watch_loss)))
        # evaluate
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    torch.save(model.state_dict(), "model_work3.pt")
    return

def predict(str):
    vocab = build_vocab()
    # 确保所有字符都在词汇表中，不在时使用 'unk'（如果已定义）
    x = [vocab.get(char, vocab.get('unk')) if char in vocab else vocab['unk'] for char in str]
    x = torch.LongTensor([x])  # 确保 x 是一个张量

    sentence_length = 4  # 确保这与训练时使用的 sentence_length 匹配
    hidden_size = 100  # 确保这与训练时使用的 hidden_size 匹配
    model = TorchModel(sentence_length, vocab, hidden_size)
    model.load_state_dict(torch.load("model_work3.pt"))
    model.eval()
    with torch.no_grad():
        result = model(x)
        predicted_class = torch.argmax(result, dim=1).item()
    print("预测类别：", predicted_class)


if __name__ == '__main__':
    main()
    predict("XabxyYzaba")
