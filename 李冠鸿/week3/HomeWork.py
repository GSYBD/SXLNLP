import json

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random

"""

使用RNN对文本进行多分类
汉字在第几个位置就是第几类

"""

class TorchRNN(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)   #embedding层

        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)    #rnn层
        self.classify = nn.Linear(vector_dim, sentence_length+1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):

        x = self.embedding(x)
        # print(x, x.shape)

        rnn_out, hidden = self.rnn(x)
        # print(rnn_out, rnn_out.shape)
        # print(hidden, hidden.shape)
        x = rnn_out[:, -1, :]   #或者写hidden.squeeze()也是可以的，因为rnn的hidden就是最后一个位置的输出
        # print(x, x.shape)

        #接线性层做分类
        y_pred = self.classify(x)
        print(y, y_pred.shape)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

# 为每个字生成一个标号
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"    #字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index+1
    vocab["unk"] = len(vocab)       #27
    vocab["你"] = len(vocab)       #28
    return vocab

def str_to_sequence(string, vocab):
    return [vocab[s] for s in string]

#随机生成一个样本
def build_sample(vocab, sentence_length):
    #随机从子表中选择sentence_length个字，可能重复
    sample_list = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if "你" not in sample_list:
        index1 = random.randint(0, sentence_length-1)
        sample_list[index1] = "你"
        y = index1
    else:
        y = sample_list.index("你")
    sample_list = str_to_sequence(sample_list, vocab)
    return sample_list, y

#建立数据集，批量生成样本
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        a, b = build_sample(vocab, sentence_length)
        dataset_x.append(a)
        dataset_y.append(b)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchRNN(char_dim, sentence_length, vocab)
    return model

#测试代码
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)     #建立200个用于测试的样本
    print("本次预测集中共有%d个样本" % (len(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)   #模型预测
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d，正确率：%f" % (correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def main():
    #配置参数
    epoch_num = 20          #训练轮数
    batch_size = 40         #每次训练样本个数
    train_sample = 1000     #每轮训练总共训练的样本总数
    char_dim = 30           #每个字的维度
    sentence_length = 10    #样本文本长度
    learning_rate = 0.001   #学习率
    #建立子表
    vocab = build_vocab()
    #建立模型
    model = build_model(vocab, char_dim, sentence_length)
    #选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    #训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)    #构造一组训练样本
            optim.zero_grad()   #梯度归零
            loss = model(x, y)  #计算loss
            loss.backward()     #计算梯度
            optim.step()        #更新权重
        print("======\n第%d轮平均loss：%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    #保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 30   #每个字的维度
    sentence_length = 10    #样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))   #加载字符表
    model = build_model(vocab, char_dim, sentence_length)       #建立模型
    model.load_state_dict(torch.load(model_path))               #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])        #将输入序列化
    model.eval()    #测试模式
    with torch.no_grad():   #不计算梯度
        result = model.forward(torch.LongTensor(x))     #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s，预测类别：%s，概率值：%s" % (input_string, torch.argmax(result[i]), result[i]))  #打印结果





if __name__ == '__main__':
    main()
    test_strings = ["adfab你sdyh", "adf你dferta", "df你wjudfwe", "e你rguytghs"]
    predict("model.pth", "vocab.json", test_strings)
