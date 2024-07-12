"""
    指定字符串中必定出现特定字符，寻找特定字符位置，判断类别
"""
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json

# 字符集合
char_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789你好世界努力学好人工智能自信走向未来"  # 字符集


class rnn_model(nn.Module):
    def __init__(self, char_dim, sentence_length, vocab, *args, **kwargs):
        super(rnn_model, self).__init__()
        self.embedding = nn.Embedding(len(vocab), char_dim)
        # self.pool = nn.AvgPool1d(sentence_length)
        self.rnn = nn.RNN(input_size=char_dim, hidden_size=sentence_length, bias=True, batch_first=True)
        # self.linear = nn.Linear(sentence_length, sentence_length)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)
        x, h = self.rnn(x)
        # x = x.transpose(1, 2)
        # x = self.pool(x)
        y_pre = h.squeeze()
        # y_pre = self.linear(h)
        if y is not None:
            return self.cross_entropy(y_pre, y)
        else:
            return y_pre


# 随机生成指定长度字符串，并且包含指定字符
def get_string(length, required_chars):
    # 从 char_list 中去掉 required_chars
    remaining_chars = [c for c in char_list if c not in required_chars]
    # 将必须包含的字符转换为列表
    result = list(required_chars)
    # 计算还需要填充的字符数量
    remaining_length = length - len(required_chars)
    # 从字符集合中随机选择剩余长度的字符，并添加到结果列表中
    result.extend(random.sample(remaining_chars, remaining_length))
    #  随机打乱结果列表中的字符顺序
    random.shuffle(result)
    # 将字符列表转换为字符串并返回
    return ''.join(result)


# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
def build_vocab():
    vocab = {"pad": 0}
    for index, char in enumerate(char_list):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab


# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 反之为负样本
def build_sample(vocab, sentence_length, chars):
    # 随机从字表选取sentence_length个字，可能重复
    x = get_string(sentence_length, chars)
    # 指定哪些字出现时为正样本
    y_index = [i for i, char in enumerate(x) if char == chars]
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    y = np.zeros(len(x))
    y[y_index] = 1
    return x, y


def build_dataset(sample_length, vocab, sentence_length, chars):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length, chars)
        # print(x)
        # print(y)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(np.array(dataset_x)), torch.FloatTensor(np.array(dataset_y))


# 找到一的位置
def find_one_position(vector):
    for i, value in enumerate(vector):
        if value == 1:
            return i


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length, chars):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length, chars)  # 建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            # print(y_p)
            # print(y_t)
            one_index = find_one_position(y_t)
            if y_p[one_index] > 0.5 and int(y_t[one_index]) == 1:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 15  # 每个字的维度
    lr = 0.005  # 学习率
    sentence_length = int(input("请输入样本文本长度："))  # 样本文本长度
    print("请选择下面字符集合中的一个作为目标字符" + char_list)
    chars = input("请输入：")
    vocab = build_vocab()
    # 建立模型
    model = rnn_model(char_dim, sentence_length, vocab)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_no in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length, chars)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length, chars)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
        # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


if __name__ == "__main__":
    main()
