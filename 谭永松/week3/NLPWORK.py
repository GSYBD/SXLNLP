# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中“你”所在的索引，索引是几就是第几类

"""


class TorchModel(nn.Module):
    # def __init__(self, vector_dim, sentence_length, vocab):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 添加batch_first=True以便批次维度在最前面
        self.fc = nn.Linear(hidden_dim, output_dim)
        # 移除sigmoid激活函数，因为CrossEntropyLoss不需要它
        self.loss = nn.CrossEntropyLoss()  # 损失函数，在训练循环中使用
        # self.activation = torch.sigmoid  # sigmoid归一化函数
        # self.loss = nn.functional.mse_loss

        # 当输入真实标签，返回loss值；无真实标签，返回预测值

    def forward(self, x, y=None):
        # x的形状应该是[batch_size, seq_length]，其中seq_length是句子的长度
        x = self.embedding(x)  # [batch_size, seq_length] -> [batch_size, seq_length, embedding_dim]
        lstm_out, (hidden, cell) = self.lstm(x)  # LSTM输出和隐藏状态
        # 取LSTM最后一个时间步的输出（也可以选择其他策略，如取平均等）
        last_hidden = hidden[-1]  # 取双向LSTM的最后一个隐藏状态（如果是双向的话，需要调整）
        # 如果是单向LSTM，可以直接使用hidden[-1, :, :]或简写为hidden[-1]
        y_pred = self.fc(last_hidden)  # [batch_size, hidden_dim] -> [batch_size, output_dim]
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            # 推理时：应用softmax获取概率分布
            y_prob = torch.nn.functional.softmax(y_pred, dim=1)  # 在类别维度上应用softmax
            return y_prob  # 返回概率分布  # 输出预测结果


# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
def build_vocab():
    chars = "南昌站年初你"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 26
    return vocab


# 随机生成一个样本
# 从所有字中选取sentence_length个字
def build_sample(vocab):
    text = "南昌站年初"
    index_to_insert = random.randint(0, len(text))
    x = text[:index_to_insert] + "你" + text[index_to_insert:]
    y = x.index("你")
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y


# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length, output_dim):
    model = TorchModel(len(vocab), char_dim, sentence_length, output_dim)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    print("本次预测集中共有%d个%d类" % ((y == 0).sum().item(), 0))
    print("本次预测集中共有%d个%d类" % ((y == 1).sum().item(), 1))
    print("本次预测集中共有%d个%d类" % ((y == 2).sum().item(), 2))
    print("本次预测集中共有%d个%d类" % ((y == 3).sum().item(), 3))
    print("本次预测集中共有%d个%d类" % ((y == 4).sum().item(), 4))
    print("本次预测集中共有%d个%d类" % ((y == 5).sum().item(), 5))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        _, predicted_indices = y_pred.max(dim=1)
        for pre,y_t in zip(predicted_indices,y):
            if pre == y_t:  # 这里使用y_t的值直接与索引比较
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
    char_dim = 20  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立字表
    vocab = build_vocab()
    embedding_dim = 64
    hidden_dim = 128
    output_dim = 6
    # 建立模型
    model = build_model(vocab, embedding_dim, hidden_dim, output_dim)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
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


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    hidden_dim = 128
    output_dim = 6
    embedding_dim = 64
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, embedding_dim, hidden_dim, output_dim)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        # 获取预测概率最高的类别的索引
        _, predicted_class = torch.max(result[i], 0)
        # 获取该类别的概率值
        predicted_prob = result[i][predicted_class]
        # 打印结果
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, predicted_class.item(), predicted_prob.item()))


if __name__ == "__main__":
    main()
    test_strings = ["南昌你站年初", "南昌站年你初", "你南昌站年初", "南昌站你年初"]
    predict("model.pth", "vocab.json", test_strings)
