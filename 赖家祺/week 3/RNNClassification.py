# coding:utf8
import json
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""
自己设计文本任务目标，使用rnn进行多分类

四个字的文本样本，"红"字第几个位置就是第几类
没有红字或者红字在首位都是0类

批大小 5
文本长度 4
特征维度 10  --->  6 --->  4
"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)

        # 第一种
        # self.pool = nn.AvgPool1d(4)  # nn.AvgPool1d(sentence_length)
        # self.classify = nn.Linear(10, 4)  # pooling

        # 第二种
        # self.rnn = nn.RNN(vector_dim, 6, bias=False, batch_first=True)
        # self.classify = nn.Linear(6, 4)

        # 第三种
        self.rnn = nn.RNN(vector_dim, 6, bias=False, batch_first=True)
        self.pool = nn.AvgPool1d(4)
        self.classify = nn.Linear(6, 4)

        # ------------------------

        self.loss = nn.functional.cross_entropy

    def forward(self, x, y_t=None):
        x = self.embedding(x)  # torch.Size([5, 4, 10])

        # 第一种, 接pooling
        # x = x.transpose(1, 2)  # torch.Size([5, 10, 4])
        # x = self.pool(x)  # torch.Size([5, 10, 1])
        # x = x.squeeze()  # torch.Size([5, 10])

        # 第二种，接RNN
        # x, hidden = self.rnn(x)  # x= torch.Size([5, 4, 6]) # hidden= torch.Size([1, 5, 6])
        # x = x[:, -1, :]   # 或者 x= hidden.squeeze()
        # # x的torch.Size 应该要变成 5, 1, 6
        # # print("x=", x, x.shape)  # torch.Size([5, 6])

        # 第三种，接RNN + pooling
        x, hidden = self.rnn(x)  # torch.Size([5, 4, 6])
        x = x.transpose(1, 2)  # torch.Size([5, 6, 4])
        x = self.pool(x)  # torch.Size([5, 6, 1])
        x = x.squeeze()  # torch.Size([5, 6])

        # 最后接入线性层
        y_p = self.classify(x)
        if y_t is not None:
            # print("y_p=", y_p)
            # print("y_t=", y_t)
            return self.loss(y_p, y_t)

        return y_p


def build_vocab():
    chars = "红橙黄绿蓝靛紫黑灰白"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1

    vocab['unk'] = len(vocab)
    return vocab


def build_model(vocab, char_dim):
    model = TorchModel(char_dim, vocab)
    return model


def build_sample(vocab, sentence_length):
    # x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    x = random.sample(list(vocab.keys()), sentence_length)  # 不放回的采样
    # print(x)  # ['unk', '蓝', '橙', '黑']
    # 没有红字或者红字在首位都是0类
    if "红" in x:
        y = x.index("红")
    else:
        y = 0

    x = [vocab.get(word, vocab["unk"]) for word in x]  # 将字转换成序号，为了做embedding
    # print(x)  # [11, 5, 2, 8]
    # x = np.array(x)

    return x, y


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        # dataset_y.append([y])
        dataset_y.append(y)

    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 测试每轮训练的准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model.forward(x)
        for y_p, y_t in zip(y_pred, y):
            # y_p已经是数值化后的 概率值
            if torch.argmax(y_p) == y_t:  # y_p = [ 0.2255, -0.6270, -0.0809,  0.2479]
                correct += 1
            else:
                wrong += 1

    correct_ptg = correct / (correct + wrong)
    print("预测正确率: %f" % correct_ptg)

    return correct_ptg


def main():
    epoch_num = 20
    batch_size = 5
    train_sample = 500
    char_dim = 10  # 每个字的维度
    sentence_length = 4  # 样本文本长度
    learning_rate = 0.005

    # 建立字表
    vocab = build_vocab()

    # 建立模型
    model = build_model(vocab, char_dim)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []  # 用来画图
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            loss = model.forward(x, y)  # model.forward(x, y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 批次内梯度归零
            watch_loss.append(loss.item())
        avg_loss = np.mean(watch_loss)
        print("\n第%d轮平均loss:%f" % (epoch + 1, avg_loss))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, avg_loss])

    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), "myRnnModel.pth")

    with open("myVocab.json", "w", encoding="utf-8") as f:
        vocab = json.dumps(vocab, ensure_ascii=False, indent=2)
        f.write(vocab)

    return


def predict(model_path, vocab_path, input_strings):
    char_dim = 10
    vocab = open(vocab_path, "r", encoding="utf-8")
    vocab = json.load(vocab)
    model = build_model(vocab, char_dim)
    model.load_state_dict(torch.load(model_path))

    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])

    model.eval()
    with torch.no_grad():
        res = model.forward(torch.LongTensor(x))

    # print(type(res), res)
    for i, input_string in enumerate(input_strings):
        print(f"输入: {input_string}, 预测类别: {torch.argmax(res[i])}, 概率值: {res[i]}")


if __name__ == '__main__':
    main()
    # vocab = {
    #     # "pad": 0,
    #     "红": 1,
    #     "橙": 2,
    #     "黄": 3,
    #     "绿": 4,
    #     "蓝": 5,
    #     "靛": 6,
    #     "紫": 7,
    #     "黑": 8,
    #     "灰": 9,
    #     "白": 10,
    #     # "unk": 11
    # }
    # test_strings = ["".join(random.sample(list(vocab.keys()), 4)) for _ in range(5)]
    # print(test_strings)
    test_strings = ['紫橙白黄', '红黄橙绿', '紫黄红蓝', '橙绿蓝红', '灰红橙黄']
    predict("myRnnModel.pth", "myVocab.json", test_strings)
