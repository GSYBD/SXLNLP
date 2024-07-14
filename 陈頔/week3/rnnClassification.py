#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
用交叉熵和rnn的多分类任务，当字符"m"，出现在字符串的第几位，就分为第几类。
如果在字符串中找不到，则分为第0类。
如果能找到且在索引第0位，分为第一类，在索引第1位，分为第二类，以此类推。

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        #embedding层,vocab是需要几个向量，词表有多大就需要多少个，vector_dim指的是多少维
        self.embedding = nn.Embedding(len(vocab), vector_dim) 
        #可以自行尝试切换使用rnn或pooling
        # self.pool = nn.AvgPool1d(sentence_length)   #池化层
        self.rnn = nn.RNN( vector_dim, vector_dim, bias=False, batch_first=True)  # RNN层
         #线性层，形状保持一致,模型输出层的输出的是类别的数量（因不重复所以是文本长度），+1是还有找不到时候第0类的存在
        self.classify = nn.Linear(vector_dim, sentence_length + 1)    
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失，pytorch的交叉熵函数自带softmax，所以不用再激活函数

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x) 
        # print("www", x)          
        #使用pooling的情况
        # x = x.transpose(1, 2)           
        # x = self.pool(x)                
        # x = x.squeeze()   
        #使用rnn的情况                       #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x, _ = self.rnn(x) # 将输入张量 x 传入RNN层，并获得RNN在整个序列上的输出 x，但忽略最后一个时间步的隐藏状态。
        x = x[:, -1, :]  # 取最后一个时间步的输出，来进行最终的预测或分类，因为它通常包含了整个序列的信息
        #接线性层做分类                         #(batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，不能重复
    x = random.sample(list(vocab.keys()), sentence_length)
    #根据"m"的位置确认类别
    if 'm' in x:
        y = x.index('m') + 1  # 类别从1开始
    #如果其中不包含"m"否则为0
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
        #y使用交叉熵损失的话就不用加括号，使用mse的话就需要加括号
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)
#调用 PyTorch 的 embedding 函数时，输入张量的数据类型期望为整数。
#cross_entropy函数也要求目标标签y（target）是整数类型。

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
    # print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y)))
    print("本次预测集中共有%d个样本"%(len(y)))
    print("样本", x, y)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        print("预测结果", y_pred)
        print("结果", y)
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1   #判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 20        #训练轮数
    batch_size = 20       #每次训练样本个数
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
    model.load_state_dict(torch.load(model_path))         #加载训练好的权重，这样我的模型跟前面训练完的文件权重就一致了
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
        print("aaadfdfgfd", result)
    for i, input_string in enumerate(input_strings): #i是索引
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, torch.argmax(result[i]), result[i].max().item())) #打印结果



if __name__ == "__main__":
    main()
    test_strings = ["fmnfee", "mzsdfg", "rqwdeg", "naqmww"]
    predict("model.pth", "vocab.json", test_strings)
