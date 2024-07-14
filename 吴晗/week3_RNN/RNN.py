#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class RNNClassifier(nn.Module):
    def __init__(self, sentence_length, embedding_dim, vocab_size, hidden_size, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, bias=False, batch_first=True)
        self.pool = nn.MaxPool1d(sentence_length)
        self.fc = nn.Linear(hidden_size, output_dim)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)
        x, h = self.rnn(x)
        x = self.pool(x.transpose(1, 2)).squeeze()
        y_pred = self.fc(x)
        # print(y_pred, y)

        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #25
    return vocab


# 根据生成字符串中a的位置确定类别
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    keys_withou_a = vocab.keys() - {'a'}
    x = [random.choice(list(keys_withou_a)) for _ in range(sentence_length - 1)]
    # 指定a的位置为类别
    insert_pos = random.randint(0, sentence_length - 1)
    x.insert(insert_pos, 'a')
    y = insert_pos
    # print(x, y)
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    # print(x, y)
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


#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)   #建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    embedding_dim = 20
    output_dim = 6
    hidden_size = 16
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    vocab_size = len(vocab)
    model = RNNClassifier(sentence_length, embedding_dim, vocab_size, hidden_size, output_dim)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        res = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        # print(torch.max(res, dim=1))
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, torch.argmax(res, dim=1)[i].item(), max(res[i].tolist()))) #打印结果



def main():
    #配置参数
    embedding_dim = 20
    output_dim = 6
    hidden_size = 16
    epoch_num = 20        #训练轮数
    batch_size = 40       #每次训练样本个数
    train_sample = 1000    #每轮训练总共训练的样本总数
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    vocab_size = len(vocab)
    build_sample(vocab, sentence_length)
    # 建立模型
    model = RNNClassifier(sentence_length, embedding_dim, vocab_size, hidden_size, output_dim)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample, vocab, sentence_length)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
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

if __name__ == "__main__":
    main()
    test_strings = ["asdwfw", "safdfg", "rqadeg", "nqkwaw", "sdoiga", "sdfgdf"]
    predict("model.pt", 'vocab.json', test_strings)
