# -*- coding: utf-8 -*-
"""
author: Chris Hu
date: 2024/7/4
desc: 通过RNN预测分类：给定一串文本，"你"字所在的索引代表了分为哪一类（下标做了调整，从1开始）
sample: 你不问过去莫谈未来我是谁又与何干, 分类为0
        不你问过去莫谈未来我是谁又与何干, 分类为1
implement: 该脚本为了快速实现模型训练，固定了测试输入是由"不问过去莫谈未来我是谁又与你何干"这个字串进行随机打乱来生成的，
           因此大大简化了场景
"""
import random
import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')  # 兼容性处理


# 定义CNN模型
class RNNClassifierModel(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size):
        super(RNNClassifierModel, self).__init__()
        self.emb = nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.rnn = nn.RNN(input_size, hidden_size, bias=False, batch_first=True)
        self.linear = nn.Linear(hidden_size, hidden_size + 1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x, h = self.rnn(self.emb(x))
        y_pred = self.linear(x[:, -1, :])
        if y is not None:
            return self.loss(y_pred, y)
        return y_pred


def shuffle(string):
    str_list = list(string)
    random.shuffle(str_list)
    return ''.join(str_list)


def build_sample(vocab, initial_str):
    target_char = "你"

    # 使用initial_str随机生成样本，简化数据长短不一以及"你"不存在的情况
    sample = shuffle(initial_str)
    # print(sample)

    x = [vocab.get(i, -1) for i in sample]
    target_index = sample.find(target_char)  # 找到即为对应的index，否则为-1
    y = [0 for _ in range(len(initial_str) + 1)]
    y[target_index] = 1
    return x, y


def create_vocab(initial_str):
    vocab = {char: index + 1 for index, char in enumerate(initial_str)}
    vocab["pad"] = 0
    vocab["unk"] = len(vocab)
    return vocab


def load_existing_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_sample_set(initial_str, sample_size):
    vocab = create_vocab(initial_str)

    # 保存训练的词表
    with open("vocabulary.json", "w") as f:
        f.write(json.dumps(vocab, ensure_ascii=False, indent=2))

    matrix_x = []
    matrix_y = []
    for _ in range(sample_size):
        x, y = build_sample(vocab, initial_str)
        matrix_x.append(x)
        matrix_y.append(y)
    return torch.LongTensor(np.array(matrix_x)), torch.FloatTensor(np.array(matrix_y))


def evaluate(model, initial_str, sample_size):
    model.eval()
    x, y = build_sample_set(initial_str, sample_size)
    wrong, correct = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if np.argmax(y_p) == np.argmax(y_t):
                correct += 1
            else:
                wrong += 1
    print(f"样本总数为：{sample_size}，正确的样本数为：{correct:d}，正确率为：{correct / sample_size}")
    # print(f"样本总数为：{sample_size}，正确的样本数为：{correct:d}，正确率为：{correct / (correct + wrong)}")
    return correct / sample_size


def main():
    initial_str = "不问过去莫谈未来我是谁又与你何干"
    epoch_num = 40
    batch_size = 20
    sample_total_size = 1000
    char_dim = 20
    hidden_size = len(initial_str)
    lr = 0.001
    vocab = create_vocab(initial_str)
    model = RNNClassifierModel(len(vocab), char_dim, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log = []
    model.train()
    print(f"训练开始，总训练样本数{batch_size * epoch_num}，每轮样本数{batch_size}，共{epoch_num}轮")
    for epoch in range(epoch_num):
        watch_loss = []
        for _ in range(int(sample_total_size / batch_size)):
            inputs, labels = build_sample_set(initial_str, batch_size)
            loss = model(inputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        print(f"Epoch {epoch + 1}/{epoch_num}, Average Loss: {np.mean(watch_loss)}")
        acc = evaluate(model, initial_str, batch_size)
        log.append([acc, float(np.mean(watch_loss))])
    torch.save(model.state_dict(), "RNNClassifierModel.pt")
    plt.plot([i for i in range(len(log))], [acl[0] for acl in log], label="acc")  # 画acc曲线
    plt.plot([i for i in range(len(log))], [acl[1] for acl in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    print("Training complete.")


# 测试模型
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度

    # 加载模型训练时的字符表
    vocab = load_existing_vocab(vocab_path)
    hidden_size = len(input_strings[0])
    model = RNNClassifierModel(len(vocab), char_dim, hidden_size)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        # 支持添加训练时未出现的字符
        x.append([vocab.get(char, vocab["unk"]) for char in input_string])
    # 开启测试模式
    model.eval()

    # 开启无梯度计算上下文
    with torch.no_grad():
        result = model(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d" % (input_string, np.argmax(result[i]) + 1))


if __name__ == '__main__':
    main()
    test_string = "不问过去莫谈未来我是谁又与你何干"
    test_strings = [shuffle(test_string) for _ in range(4)]
    test_strings.append("abcde你听说很好fghijk")
    predict("RNNClassifierModel.pt", "vocabulary.json", test_strings)
