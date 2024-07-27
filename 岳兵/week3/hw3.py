#coding:utf8
import random
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
手动实现简单的神经网络
使用pytorch实现RNN
手动实现RNN
对比
任务目标：在字符串中寻找a的位置，a可以不存在（单独一类）
注意体会pooling方法和rnn方法的区别
"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        #self.pool = nn.AvgPool1d(sentence_length)
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first = True)

        self.classify = nn.Linear(vector_dim, sentence_length + 1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x,y=None):
        x = self.embedding(x)
        rnn_out, hidden = self.rnn(x)
        x = rnn_out[:, -1, :]

        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

#建立一个字符集
def build_vocab():
    chars = "abcdefghijk"
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1
    vocab['unk'] = len(vocab)
    return vocab

#随机生成一个样本
def build_sample(vocab, sentence_length):
    x = random.sample(list(vocab.keys()), sentence_length)
    if "a" in x:
        y = x.index("a")
    else:
        y = sentence_length
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

#建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x),torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    print("本次测试集中共有%d个样本"%(len(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    epoch_num = 20
    batch_size = 40
    train_sample = 1000
    char_dim = 30
    sentence_length = 10
    learning_rate = 0.001

    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length)
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("==========\n第%d轮平均loss:%f" %(epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])

    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), "model.pth")

    writer = open("vocab.json", "w", encoding = "utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return
def predict(model_path, vocab_path, input_strings):
    char_dim = 30
    sentence_length = 10
    vocab = json.load(open(vocab_path, "r",encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类型：%s, 概率值：%s" %(input_string, torch.argmax(result[i]), result[i]))

if __name__ == "__main__":
    main()
    test_strings = ["kijabcdefh", "gijkbcdeaf", "gkijadfbec", "kijhdefacb"]
    predict("model.pth", "vocab.json", test_strings)







