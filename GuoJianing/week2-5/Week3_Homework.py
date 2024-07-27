# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt






class TorchModel(nn.Module):
    def __init__(self, input_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab)+1, input_dim, padding_idx=0)
        self.rnn = nn.RNN(input_dim, input_dim, batch_first=True)
        self.layer = nn.Linear(input_dim, sentence_length+1)
        self.loss = nn.functional.cross_entropy  

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  
        rnn_out, _ = self.rnn(x)  
        x = rnn_out[:,-1,:]
        y_pred = self.layer(x) 
        if y is not None:
            return self.loss(y_pred, y.squeeze())
        else:
            return y_pred



def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  
    vocab['unk'] = len(vocab) + 1
    return vocab


def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]

    if "a" in x:
        y = x.index("a")
    else:
        y = 6
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    total = 200  # 测试样本数量
    x, y = build_dataset(total, vocab, sample_length)  
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1  
            else:
                wrong += 1
    print("正确预测个数：%d / %d, 正确率：%f" % (correct, total, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num = 15  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = build_vocab()  # 建立字表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    optim = torch.optim.Adam(model.parameters(), lr=0.005)  # 建立优化器
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构建一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    torch.save(model.state_dict(), "model.pth")
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 最终预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式，不使用dropout
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        if torch.argmax(result[i]) == 6:
            x = "负样本"
        else:
            x = "正样本"
        print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, x, result[i])) #打印结果


if __name__ == "__main__":
    main()
    test_strings = ["juvxee", "yrwfrg", "rbwaqg", "nahdww"]
    predict("model.pth", "vocab.json", test_strings)
