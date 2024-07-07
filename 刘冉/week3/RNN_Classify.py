# coding: utf-8

import torch
import torch.nn as nn
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import os

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务 五分类任务
判断文本中出现文字进行分类：
    出现"八" or "斗" 则为第一类
    出现"精" or "品" 则为第二类
    出现"学" or "习" 则为第三类
    出现"工" or "作" 则为第四类
    其他为第五类, 如果同时出现，则第一个出现的为最终结果

"""

class ClassifyModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, hidden_size, vocab):
        super(ClassifyModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.rnn = nn.RNN(vector_dim, hidden_size, bias=False, batch_first=True)
        self.pool = nn.AvgPool1d(sentence_length)
        self.classify = nn.Linear(hidden_size, 5) # 预测5分类任务
        self.activation = torch.softmax # 激活函数
        self.loss = nn.CrossEntropyLoss()

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x) # shape: (batch_size, seq_len, vector_dim)
        # RNN类的模型会同时返回隐单元向量，我们只取序列结果
        x = self.rnn(x)[0]  # shape: (batch_size, seq_len, hidden_size)
        x = self.pool(x.transpose(1, 2)).squeeze() # shape: (batch_size, hidden_size)
        x = self.classify(x) # shape: (batch_size, 5)
        y_pred = self.activation(x, dim=1) # shape: (batch_size,5)
        if y is not None:
            return self.loss(y_pred, y.squeeze())
        else:
            return y_pred
    

def get_vocab():
    vocab = json.load(open("text_vocab.json", "r", encoding="utf8")) #加载字符表
    return vocab

# 随机生成一个样本， 从所有字中选取sentence_length个字
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，分别得到字和下标 可能重复
    sentence = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    x = [vocab[i] for i in sentence]
    #按条件分类  第一个出现的为最终结果
    for string in sentence:
        if string in "八斗":
            return x, 0
        elif string in "精品":
            return  x, 1
        elif string in "学习":
            return x, 2
        elif string in "工作":
            return x, 3
    return x, 4

# 建立数据集  需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 建立模型
def build_model(vocab, char_dim, sentence_length):
    hidden_size = 10 # 隐藏层神经元个数
    model = ClassifyModel(char_dim, sentence_length, hidden_size, vocab)
    return model

# 测试代码  测试模型的准确率

def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x) # 模型预测
        for y_p, y_t in zip(y_pred, y):
            if y_p.argmax(dim=0) == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：{}， 正确率：{}".format(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    # 配置参数
    epoch_num = 40 # 训练轮数
    batch_size = 20 # 每次训练样本个数
    train_sample = 8000 # 每轮训练总共训练的样本数
    char_dim = 20    # 每个字的维度
    sentence_length = 8 # 句子长度
    vocab = get_vocab()
    learning_rate = 0.005 # 学习率
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()   # 梯度清零
            loss = model(x, y) # 计算loss
            loss.backward() # 计算梯度
            optim.step() # 权重更新
            watch_loss.append(loss.item())
        print("==========第%d轮训练==========" % (epoch + 1))
        acc = evaluate(model, vocab, sentence_length)   # 训练集准确率
        log.append([acc, np.mean(watch_loss)])
        print("第%d轮平均loss:%f,last_loss:%f,acc:%f" % (epoch + 1, np.mean(watch_loss), watch_loss[-1], acc))
    # 画图
    plt.plot(range(len(log)), [i[0] for i in log], label="acc")
    plt.plot(range(len(log)), [i[1] for i in log], label="loss")
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    return

# 使用训练好的模型预测
def predict(model_path, vocab, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 8  # 样本文本长度
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab['[unk]']) for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    print(x)
    with torch.no_grad():  #不计算梯度
        y_pred = model(torch.LongTensor(x))  #模型预测

    for input_string, result in zip(input_strings, y_pred):
        print("输入：%s, 预测类别为第%d类, " % (input_string, result.argmax(dim=0) + 1))


if __name__ == "__main__":
    main()
    vocab = get_vocab()
    test_strings = ["你并经发斗好更子", "才出精八真个动真",
                    "后时并作精更下事", "品次不起样成你子",
                    "以么多学得成也手", "说经工习得学服起",
                    "生但看年外道当又", "后以为向的和对而",
                    "心样工正你动主这", "儿知者作然向成大"]
    predict("model.pth", vocab, test_strings)
