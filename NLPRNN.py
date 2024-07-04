#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中特定字符出现在第几位，则为几分类任务

"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层
        # self.classify = nn.RNN(vector_dim, sentence_length + 1, bias=False, batch_first=True)  # RNN
        self.lstm = nn.LSTM(vector_dim, 128, bias=False, batch_first=True)  # RNN
        self.classify = nn.Linear(128, sentence_length+1)     #线性层
        self.loss = torch.nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)  # LSTM的输出和隐藏状态，我们只关心输出
        # 取LSTM最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]  # (batch_size, 128)
        y_pred = self.classify(lstm_out)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  #输出预测结果              #输出预测结果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # print(x)
    # 检查'a'在x中的位置（实际上是检查x中哪个索引对应vocab中的'a'）
    a_index_in_vocab = vocab.get('a')  # 获取'a'在vocab中的索引
    # print(a_index_in_vocab)
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    a_position_in_x = next((i for i, idx in enumerate(x) if idx == a_index_in_vocab), None)  # 查找这个索引在x中的位置
    # print(a_position_in_x)

    if a_position_in_x is None:
        # 如果没有找到'a'，则y设置为sentence_length或其他您选择的默认值
        y = sentence_length
    else:
        # 如果找到了'a'，则y等于'a'在x中的位置
        y = a_position_in_x
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
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    # print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 200 - sum(y)))
    # print(y)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        _, preds = torch.max(y_pred, 1)  # 获取预测的最高概率的索引
        corrects = (preds == y).float()  # 比较预测和真实标签
        # print("原值：", y)
        # print("预测值：", preds)
        # print("对比值：", corrects)
        for i in range(corrects.size(0)):  # 与真实标签进行对比
            if corrects[i] == 1:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/float(200)))
    return correct/float(200)


def main():
    #配置参数
    epoch_num = 20        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.001  #学习率
    # 建立字表
    vocab = build_vocab()
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
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  #画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  #画loss曲线
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
        _, preds = torch.max(result, 1)  # 获取预测的最高概率的索引
        print(preds)
    for i, input_string in enumerate(input_strings):
        print(preds[i]) #打印结果



if __name__ == "__main__":
    main()
    # test_strings = ["anvfee", "azsdfg", "rqadeg", "nakwww"]
    # predict("model.pth", "vocab.json", test_strings)
