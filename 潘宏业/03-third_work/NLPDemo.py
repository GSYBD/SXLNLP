import torch
import torch.nn as nn
import numpy as np
import random
from matplotlib import pyplot as plt
import json

"""
随机生成10个字母，判断字母在a-m之间有多少个，去除[PAD]和[UNK]
"""

class TorchModel(nn.Module):
    def __init__(self, vocab, hidden_size, output_size, sentence_len):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.pool = nn.AvgPool1d(sentence_len)
        self.linear = nn.Linear(hidden_size, output_size)
        self.act = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x_embedding = self.embedding(x)     # (batch_size, sentence_len) -> (batch_size, sentence_len, hidden_size)
        _, x_rnn = self.rnn(x_embedding)    # (batch_szie, sentence_len, hidden_size) -> (1, batch_size, hidden_size)
        # x_pool = self.pool(x_rnn.transpose(1, 2)).squeeze()   # (batch_size, sentence_len, hidden_size) -> (batch_size, hidden_size)
        x_linear = self.linear(x_rnn.squeeze())  # (batch_size, hidden_size) -> (batch_size, output_size)
        y_pred = self.act(x_linear)  # (batch_size, output_size) -> (batch_size, output_size)
        if y is None:
            return y_pred
        else:
            return self.loss(y_pred, y)

def load_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {}
    vocab['[PAD]'] = 0
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['[UNK]'] = len(vocab)
    return vocab

def load_data(vocab, sentence_len):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_len)]
    y = 0
    for i in x:
        if i != '[UNK]' and i != '[PAD]' and ord(i) - ord('a') < 13:
            y += 1
    x = [vocab.get(i, vocab['[UNK]']) for i in x]
    return x, y

def dataset(data_num, vocab, sentence_len):
    X = []
    Y = []
    for num in range(data_num):
        x, y = load_data(vocab, sentence_len)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)

def evaluate(model, vocab, sentence_len, sample_num):
    model.eval()
    x_eval, y_eval = dataset(sample_num, vocab, sentence_len)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x_eval)
        for y_p, y_t in zip(y_pred, y_eval):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("测试样本总数为%d，其中预测正确样本个数为%d，正确率为%f" % ((correct + wrong), correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    vocab = load_vocab()
    data_num = 1000
    sentence_len = 10
    X, Y = dataset(data_num, vocab, sentence_len)

    hidden_size = 128
    output_size = 11
    model = TorchModel(vocab, hidden_size, output_size, sentence_len)
    epoch_num = 50
    batch_size = 10
    lr = 10e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    test_num = 100
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(data_num // batch_size):
            optimizer.zero_grad()
            x = X[batch * batch_size : (batch + 1) * batch_size]
            y = Y[batch * batch_size : (batch + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        loss_avg = np.mean(watch_loss)
        print("第%d轮平均损失为%f" % (epoch + 1, loss_avg))
        acc = evaluate(model, vocab, sentence_len, test_num)
        log.append([acc, loss_avg])

    torch.save(model.state_dict(), "model.pth")

    plt.plot(range(len(log)), [l[0] for l in log], label='acc')
    plt.plot(range(len(log)), [l[1] for l in log], label='loss')
    plt.show()

    writer = open("vocab.json", 'w', encoding='utf8')
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

if __name__ == '__main__':
    main()