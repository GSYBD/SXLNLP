# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
输入一个字符串，根据字符 “人” 所在位置进行分类
对比rnn和pooling做法

"""


class TorchModel(nn.Module):
    def __init__(self,  vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层
        
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        
        #不存在字的情况
        self.classify = nn.Linear(vector_dim, sentence_length + 1)
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  
        
        rnn_out, hidden = self.rnn(x)
        x = rnn_out[:, -1, :]
        
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return y_pred  # 输出预测结果


def build_vocab():
    chars = "一二三木头人四五六"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
def build_sample(vocab, sentence_length):
    #注意这里用sample，是不放回的采样，每个字母不会重复出现，但是要求字符串长度要小于词表长度
    # print("sentence_length:",sentence_length)
    # print("len(vocab):",len(vocab))
    x = random.sample(list(vocab.keys()), sentence_length)
    #指定哪些字出现时为正样本
    if "人" in x:
        y = x.index("人")
    else:
        y = sentence_length
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(sample_length, vocab, sentence_length):
    X = []
    Y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    # print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
                
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    learning_rate = 0.001  # 学习率
    char_dim = 30         #每个字的维度    
    sentence_length = 10   #样本文本长度

    # 建立字表
    vocab = build_vocab()
    
    # 建立模型
    model = TorchModel(char_dim, sentence_length, vocab)
    
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(int(train_sample / batch_size)):    
            x, y = build_dataset(batch_size, vocab, sentence_length)
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    # 保存模型
    torch.save(model.state_dict(), "model_for_rnn.pth")
    # 保存词表
    writer = open("vocab_for_rnn.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 30  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = TorchModel(char_dim,sentence_length,vocab)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化

    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, torch.argmax(result[i]), result[i]))


if __name__ == "__main__":
    # main()
    test_strings = ["六六六六人六六六六", "木头一木头人木头一", "木头木头木头木头人", "四五六四五人四五六"]
    predict("model_for_rnn.pth", "vocab_for_rnn.json", test_strings)
