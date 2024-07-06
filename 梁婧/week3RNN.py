# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import torch.nn.functional as F

'''
设计文本任务目标，使用rnn进行多分类
判断文本中字符 a 出现的次数，分为三类，0次，1次，多于1次
'''

class TorchRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim):
        super(TorchRNN, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)
        self.loss_fn = nn.functional.cross_entropy

    def forward(self, x, y=None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        # 取最后一个时间步的输出作为全连接层的输入
        last_output = output[:, -1, :]  # (batch_size, hidden_dim)
        logits = self.fc(last_output)  # (batch_size, num_classes)

        if y is not None:
            # 计算损失
            return self.loss_fn(logits, y)
        else:
            return logits  # 返回 logits

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab
    # print(type(vocab))

#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]

    # 计算'a'在文本中出现的次数
    a_count = x.count('a')

    # 根据'a'出现的次数设置标签
    if a_count == 0:
        y = 0  # 不包含'a'
    elif a_count == 1:
        y = 1  # 包含'a'恰好一次
    else:
        y = 2  # 包含'a'多于一次

    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)

    # 转换为Tensor
    dataset_x = torch.LongTensor(dataset_x)  # x已经是整数列表，适合LongTensor
    # 对于分类任务，如果使用CrossEntropyLoss，则直接使用LongTensor
    dataset_y = torch.LongTensor(dataset_y)

    return dataset_x, dataset_y

#建立模型
def build_model(vocab_size, embedding_dim, hidden_dim):
    model = TorchRNN(vocab_size, embedding_dim, hidden_dim)
    return model
#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本

    # 确保y是long类型，因为它是类别索引
    y = y.long()

    with torch.no_grad():
        y_pred = model(x)
        _, predicted = torch.max(y_pred, 1)  # 获取概率最高的类别的索引

    correct = (predicted == y).sum().item()  # 计算正确预测的个数
    total = y.size(0)  # 总样本数
    accuracy = correct / total

    print(f"正确预测个数：{correct}, 正确率：{accuracy:.4f}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    embedding_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.005  # 学习率

    # 建立字表
    vocab = build_vocab()
    # print(vocab)
    vocab_size = len(vocab)  # 计算词汇表的大小
    # print(type(vocab_size))
    model = build_model(vocab_size, embedding_dim, sentence_length)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 假设 model 的 forward 方法已经能够处理 y 并返回损失
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        # 评估模型（这里假设 evaluate 函数能够正确工作）
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        print(f"第{epoch + 1}轮准确率: {acc}")
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    embedding_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    vocab_size = len(vocab)
    model = build_model(vocab_size, embedding_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测

    _, predicted = torch.max(result, 1)  # 找到概率最高的类别的索引

    for i, input_string in enumerate(input_strings):
        # 注意：这里我们只打印了预测类别的索引，如果需要概率值，可以单独提取
        predicted_class = predicted[i].item()  # 转换为 Python 整数
        # 如果需要打印概率值，可以这样做：
        predicted_prob = result[i, predicted_class].item()  # 提取最高概率值
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string[:sentence_length], predicted_class, predicted_prob))

if __name__ == "__main__":
    main()
    test_strings = ["fnvfee", "wzsdfg", "rqwdeg", "nakwww", "nakwaw"]
    predict("model.pth", "vocab.json", test_strings)