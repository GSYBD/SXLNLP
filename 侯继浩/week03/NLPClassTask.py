#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import string
from collections import Counter

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp分类任务
判断文本，以a开头则为分类1，以a结尾则为分类3，其他则为分类2

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层
        self.pool = nn.AvgPool1d(sentence_length)   #池化层
        #self.classify = nn.Linear(vector_dim, 3)     #线性层
        self.classify = nn.RNN(vector_dim, 3, bias=True, batch_first=True, num_layers=2)
        self.activation = torch.sigmoid     #sigmoid归一化函数
        self.loss = nn.functional.cross_entropy  #loss函数采用均方差损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = x.transpose(1, 2)                      #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        x = self.pool(x)                           #(batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        x = x.squeeze()                            #(batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        x = self.classify(x)                       #(batch_size, vector_dim) -> (batch_size, 1) 3*5 5*1 -> 3*1

        y_pred = self.activation(x[0])                #(batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = string.ascii_lowercase  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, length, first='', end=''):
    #随机从字表选取sentence_length个字，可能重复
    letters = list(string.ascii_lowercase)

    if first != '':
        letters.remove(first)
        length -= 1
    if end != '':
        letters.remove(end)
        length -= 1

    random.shuffle(letters)
    res = first + ''.join(letters[:length]) + end
    #指定哪些字出现时为正样本
    if res[0] == 'a':
        y = [1, 0, 0]
    elif res[len(res)-1] == 'a':
        y = [0, 0, 1]
    else:
        y = [0, 1, 0]
    x = [vocab.get(word, vocab['unk']) for word in res]   #将字转换成序号，为了做embedding
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        random_class = random.choice([1, 2, 3])
        if random_class==1:
            first, end = 'a', ''
        elif random_class==3:
            first, end = '', 'a'
        else:
            first = end = ''
        x, y = build_sample(vocab, sentence_length, first, end)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(np.array(dataset_x)), torch.FloatTensor(np.array(dataset_y))

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

def get_class_label(x):
    labels = [1, 2, 3]
    return labels[int(x.numpy().argmax())]

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    yRes = []
    for y1 in y:
        yRes.append(int(get_class_label(y1)))
    yCounter = Counter(yRes)
    print("本次预测集中共有%d个1分类，%d个2分类，%d个3分类" % (yCounter[1], yCounter[2], yCounter[3]))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if int(get_class_label(y_p)) == int(get_class_label(y_t)):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 30        #训练轮数
    batch_size = 30       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率
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
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d" % (input_string, get_class_label(result[i]))) #打印结果

if __name__ == "__main__":
    main()
    #test_strings = ["fnvfee", "wzsdfg", "rqwdeg", "nakwww"]
    #predict("model.pth", "vocab.json", test_strings)
