import json
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 曹哲鸣第三周作业
# 对文本进行分类
# 判断文本中的‘你’在哪个位置，则该文本就为第几类
# 例：'你好么'，为第一类，‘我揍你’，为第三类

class TorchModule(nn.Module):
    def __init__(self, input_size, hidden_size, vocab):
        super(TorchModule, self).__init__()
        self.embedding = nn.Embedding(len(vocab), input_size, padding_idx=0)    #embedding层
        self.rnn = nn.RNN(input_size, hidden_size, bias=False, batch_first=True)    #rnn层
        self.linear = nn.Linear(hidden_size, hidden_size + 1)   #线性层
        self.loss = nn.functional.cross_entropy     #loss函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)               #[batch_size, text_size] -> [batch_size, text_size, input_size]
        x, h = self.rnn(x)                  #[batch_size, text_size, input_size] -> [batch_size, text_size, hidden_size]
        x = x[:, -1, :]                     #[batch_size, text_size, hidden_size] -> [batch_size, hidden_size]
        y_pred = self.linear(x)             #[batch_size, hidden_size] -> [batch_size, hidden_size+1]
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

#生成字典
def BulidVocab():
    str = "你abcdefghi"
    vocab = {"pad" : 0}
    for index, char in enumerate(str):
        vocab[char] = index + 1
    vocab["unk"] = len(vocab)
    return vocab

#生成一个样本
def BuildSample(text_size, vocab):
    x = random.sample(list(vocab.keys()), text_size)
    y = np.zeros(text_size + 1)
    if '你' in x:
        y[x.index("你")] = 1
    else:
        y[-1] = 1
    x = [vocab.get(words, vocab["unk"]) for words in x]
    return x, y

#生成数据集
def BuildDataSet(batch_size, text_size, vocab):
    X = []
    Y = []
    for i in range(batch_size):
        x, y =BuildSample(text_size, vocab)
        X.append(x)
        Y.append(y)
    return X, Y

#测试代码，测试每轮准确率
def eveluate(model, text_size, vocab):
    model.eval()
    x, y = BuildDataSet(100, text_size, vocab)
    wrong, correct = 0, 0
    with torch.no_grad():
        x = torch.LongTensor(x)
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if np.argmax(y_p) == np.argmax(y_t):
                correct += 1
            else:
                wrong += 1
        print("样本总数为：100，正确的样本数为：%d，正确率为：%f" %(correct, correct/(correct + wrong)))
        return correct/(correct + wrong)


#训练模型
def main():
    #配置参数
    epoch_size = 20
    sample_size = 1000
    batch_size = 20
    text_size = 6
    vector_dim = 20
    hidden_size = 6
    lr = 0.002
    #生成字典
    vocab = BulidVocab()
    #建立模型
    model = TorchModule(vector_dim, hidden_size, vocab)
    #优化器
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    log = []
    #训练过程
    for epoch in range(epoch_size):
        model.train()
        watch_loss = []
        for batch in range(int(sample_size / batch_size)):
            x_train, y_train = BuildDataSet(batch_size, text_size, vocab)
            x = torch.LongTensor(x_train)
            y = torch.FloatTensor(y_train)
            loss = model(x, y)      #计算loss
            loss.backward()         #计算梯度
            optim.step()            #更新权重
            optim.zero_grad()       #权重归零
            watch_loss.append(loss.item())
        print("每轮的平均loss为：%f" %(np.mean(watch_loss)))
        acc = eveluate(model, text_size, vocab)
        log.append([acc, np.mean(watch_loss)])

    #画图
    plt.plot(range(len(log)), [l[0] for l in log], label = "acc")
    plt.plot(range(len(log)), [l[1] for l in log], label = "loss")
    plt.legend()
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    vector_dim = 20  # 每个字的维度
    hidden_size = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = TorchModule(vector_dim, hidden_size, vocab)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d" % (input_string, np.argmax(result[i])+1)) #打印结果




if __name__ == "__main__":
    main()
    test_strings = ["你abcdf", "egh你ab", "fg你hab", "abcde你"]
    predict("model.pt", "vocab.json", test_strings)
