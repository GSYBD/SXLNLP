import copy

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

"""
基于pytorch的网络编写
要求：
自己设计文本任务目标，使用RNN完成一个nlp多分类任务。
设计的任务目标如下：
给定一个包含特定字符“虹“的字符串，该字符出现在字符串的第几个位置，则为第几类。
"""

class TorchRNN(nn.Module):
    def __init__(self, vector_dim, sentence_length, hidden_side, vocab):
        super(TorchRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层
        self.pool = nn.AvgPool1d(sentence_length )  #池化层
        self.layer = nn.Linear(hidden_side, sentence_length +1)  #线性层
        self.rnn = nn.RNN(vector_dim, hidden_side, bias=False, batch_first=True)  #RNN层
        self.bn1 = nn.LayerNorm(vector_dim)  #LayerNorm层
        self.bn2 = nn.LayerNorm(hidden_side)  #LayerNorm层
        # self.dp = nn.Dropout(0.1)  #Dropout层
        self.loss = nn.functional.cross_entropy  #loss函数采用交叉熵

    def forward(self, x, y=None):
        x = self.embedding(x)        #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = self.bn1(x)              #(batch_size, sen_len, vector_dim) -> (batch_size, sen_len, vector_dim)  归一化
        x, h = self.rnn(x)           #(batch_size, sen_len, vector_dim) -> (batch_size, sen_len, hidden_side)

        print(x)
        x = x.transpose(1, 2)        #(batch_size, sen_len, hidden_side) -> (batch_size, hidden_side, sen_len)
        # print(x.shape)
        x = self.pool(x)             #(batch_size, hidden_side, sen_len)->(batch_size, hidden_side, 1)
        # print(x.shape)
        x = x.squeeze()              #(batch_size, hidden_side, 1) -> (batch_size, hidden_side)
        # print(x.shape)
        # x = self.dp(x)               #(batch_size, hidden_side) -> (batch_size, hidden_side)
        # print(x.shape)
        x = self.bn2(x)              #(batch_size, hidden_side) -> (batch_size, hidden_side)  归一化
        # y_pred = self.bn2(x)
        # print(x.shape)
        y_pred = self.layer(x)       #(batch_size, hidden_side) -> (batch_size, sentence_length)
        # print(y_pred.shape)
        # print(y.shape)
        # print(y_pred)
        # print(y)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz你好人工智能NLP彩虹在那"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab

#随机生成一个包含特定字符“好“的字符串样本
#从所有字中选取sentence_length个字
def build_sample(vocab, sentence_length):
    #复制一份词表
    vocab_copy = copy.deepcopy(list(vocab.keys()))
    #去掉词表中的特定字符“好“
    vocab_copy.remove('虹')
    #随机生成一个包含(sentence_length - 1)个字符的字符串
    # x = [random.choice(vocab_copy) for _ in range(sentence_length - 1)]
    x = random.sample(list(vocab.keys()), sentence_length)

    #加上特定字符“好“
    # x.append('虹')
    #随机打乱字符串中的字符顺序
    # random.shuffle(x)
    #提取特定字符“好“的索引值+1，作为类别

    if '虹' in x:
        y = x.index('虹') + 1   # 类别从1开始
    else:
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
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
def build_model(vocab, char_dim, sentence_length, hidden_side):
    model = TorchRNN(char_dim, sentence_length, hidden_side, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    print("本次预测集中共有%d个样本" % (len(y.numpy())))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if np.argmax(y_p.numpy()) == int(y_t):
                correct += 1  #分类正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 30        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    hidden_side = 8       #RNN隐藏层维度
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length, hidden_side)
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
    torch.save(model.state_dict(), "model1.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    hidden_side = 8  # RNN隐藏层维度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length, hidden_side)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化

    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
        y_p = torch.softmax(result, dim=1)
    for input_string, res in zip(input_strings, y_p):
        print(res.numpy())
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, np.argmax(res.numpy()) , np.max(res.numpy()))) #打印结果



if __name__ == "__main__":
    main()
    print("\n固定字符串测试：")
    test_strings1 = ["es虹fee", "wzsd虹g", "p好wdeg", "folw虹w"]
    predict("model1.pth", "vocab.json", test_strings1)
