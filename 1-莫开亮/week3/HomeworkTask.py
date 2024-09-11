# -*- coding: utf-8 -*-
import json
import torch
import torch.nn as nn
import random
import numpy as np
from matplotlib import pyplot as plt

"""
基于pytorch的网络编写
实现一个网络完成一个简单nlp任务,使用rnn进行多分类任务：
按m字符在样本中的位置分类，如果样本中不存在字符m，那么就归类为字符串长度
"""


# 定义Torch模型
class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        # embedding层
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        # RNN层
        self.layer = nn.RNN(vector_dim, vector_dim, batch_first=True)
        self.classify = nn.Linear(vector_dim, sentence_length + 1)  # 线性层
        self.loss = nn.functional.cross_entropy  # 损失函数,交叉熵

    def forward(self, x, y=None):
        # 词向量
        x = self.embedding(x)
        # 经过RNN层
        rnn_out, hidden = self.layer(x)
        x = hidden.squeeze()
        # 线性层分类
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred, y)  # 输入x,y时，计算损失值
        return y_pred  # 预测值


# 建立字典
def build_vocab():
    # 建立字符字典
    chars = 'abcdefghijklmnopqrstuvwxyz'
    vocab = {"pad": 0}
    for i, char in enumerate(chars):
        vocab[char] = i + 1
    vocab["unk"] = len(vocab)
    return vocab


# 生成一个样本
def build_sample(vocab, sentence_length):
    # 随机生成一个样本
    # 从vocab中随机取sentence_length个字符，作为样本
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 指定标签,如果字符串中存在k，下标作为类别，否则类别为字符串长度
    if 'm' in x:
        y = x.index('m')
    else:
        y = sentence_length
    # 转化为索引 [10,1,23,8,3,...]
    x = [vocab.get(word, vocab["unk"]) for word in x]
    return x, y


# 生成数据集
def build_dataset(sample_length, vocab, sentence_length):
    x, y = [], []
    for i in range(sample_length):
        x1, y1 = build_sample(vocab, sentence_length)
        x.append(x1)
        y.append(y1)
    return torch.LongTensor(x), torch.LongTensor(y)


# 模型预测结果
def evaluate(model, vocab, sentence_length):
    model.eval()
    sample_length = 100  # 总样本
    x, y = build_dataset(sample_length, vocab, sentence_length)
    print(y)

    print("本次预测集中共有%d个样本" % (len(x)))
    correct, wrong = 0, 0
    with torch.no_grad():  # 测试不记录梯度信息
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20        # 训练轮数
    batch_size = 200      # 每次训练样本个数
    train_sample = 10000  # 训练样本个数
    learning_rate = 0.01  # 学习率
    char_dim = 20         # 每个字符维度
    sentence_length = 8   # 句子长度
    # 建立字符字典
    vocab = build_vocab()
    # 建立模型
    model = TorchModel(char_dim, sentence_length, vocab)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练模型
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optimizer.zero_grad()  # 梯度归零
            loss = model(x, y)     # 计算损失
            loss.backward()        # 计算梯度，误差反向传播
            optimizer.step()       # 更新权重
            watch_loss.append(loss.item())  # 记录误差
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])  # 记录结果
        print(watch_loss[-1])
        if watch_loss[-1] < 0.001:
            break
    # 保存模型
    torch.save(model.state_dict(), "model2.pth")
    # 保存词表
    with open("vocab.json", "w", encoding="utf8") as writer:
        writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

# 使用模型预测
def predict(model_path, vocab_path, input_strings):
    # 加载字符字典
    vocab = json.load(open(vocab_path, encoding="utf8"))
    print(vocab)
    char_dim = 20        # 每个字符维度,和训练时保存一致
    sentence_length = 8  # 句子长度,和训练时保存一致
    # 建立模型
    model = TorchModel(char_dim, sentence_length, vocab)
    # 加载模型
    model.load_state_dict(torch.load(model_path))
    # 将输入字符串转换为索引序列化
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])
    print(x)
    # 用0补全
    for i in range(len(x)):
        x[i] = x[i] + [0] * (sentence_length - len(x[i]))
    print(x)

    # 开始预测
    model.eval()
    with torch.no_grad():  # 测试不计算梯度信息
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s" % (
            input_string, torch.argmax(result[i])))

if __name__ == "__main__":
    main()
    # 使用预测
    test_strs = ["goods", "xmind", "mask", "name", "abcdefg", "pthmoom"]
    predict("model2.pth", "vocab.json", test_strs)
