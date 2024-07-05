import random

import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
import torch.utils.data as Data
"""
构建一个 用RNN实现的 判断某个字符的位置 的任务

5 分类任务 判断 a出现的位置 返回index +1 or -1
"""
class TorchRnnModel(nn.Module):
    def __init__(self,sentence_length, hidden_size, vocab, input_dim, output_size):
        super(TorchRnnModel,self).__init__()
        self.emb = nn.Embedding(len(vocab) + 1, input_dim)
        self.rnn = nn.RNN(input_dim, hidden_size, batch_first=True)

        self.pool = nn.MaxPool1d(sentence_length)
        self.leaner = nn.Linear(hidden_size, output_size)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.emb(x)
        x, h = self.rnn(x)
        x = self.pool(x.transpose(1, 2)).squeeze()
        y_pred = self.leaner(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred

# 创建字符集 只有6个 希望a出现的概率大点
def build_vocab():
    chars = "你好a"
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab) + 1
    return vocab

# 构建样本集
def build_dataset(vocab, data_size, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(data_size):
        x, y = build_simple(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 构建样本
def build_simple(vocab, sentence_length):
    # 随机生成 长度为4的字符串
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # if x.count('a') != 0:
    #     y = x.index('a')
    # else:
    #     y = 4
    #指定哪些字出现时为正样本
    if set('a') & set(x):
        y = 1
    else:
        #指定字都未出现，则为负样本
        y = 0
    # 转化为 数字
    x = [vocab[char] for char in list(x)]
    return x, y

def build_model(sentence_length, hidden_size, vocab, input_dim, output_size):
    model = TorchRnnModel(sentence_length, hidden_size, vocab, input_dim, output_size)
    return model
# 测试
def main():
    batch_size = 20
    simple_size = 5000
    vocab = build_vocab()
    # 每个样本的长度为4
    sentence_length = 4
    # 样本的向量维度为10
    input_dim = 10
    # rnn的隐藏层设置为20
    hidden_size = 20
    # 5 分类任务
    output_size = 5
    # 学习率
    learning_rate = 0.0001
    # 轮次
    epoch_size = 20
    model = TorchRnnModel(sentence_length, hidden_size, vocab, input_dim, output_size)

    # 优化函数
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 样本
    x, y = build_dataset(vocab, simple_size, sentence_length)
    dataset = Data.TensorDataset(x, y)
    dataiter = Data.DataLoader(dataset, batch_size)
    for epoch in range(epoch_size):
        epoch_loss = []
        model.train()
        for x, y_true in dataiter:
            loss = model(x, y_true)
            loss.backward()
            optim.step()
            optim.zero_grad()
            epoch_loss.append(loss.item())
        print("第%d轮 loss = %f" % (epoch + 1, np.mean(epoch_loss)))
        # evaluate
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc,float(np.mean(epoch_loss))])
        # 保存模型
    torch.save(model.state_dict(), "model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

#测评结果
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(vocab, 1000, sentence_length)
    correct, wrong = 0, 0
    y_pred = model(x)
    with torch.no_grad():
        for y_p, y_t in zip(y_pred, y):
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d / %d, 正确率：%f" % (correct, correct + wrong, correct / (correct + wrong)))
    return correct / (correct + wrong)
#使用训练好的模型做预测
def predict(input_strings,model_path,vocab_path):
    char_dim = 20                   #字符向量维度
    sentence_length = 6             #句子长度
    vocab = json.load(open(vocab_path,'r',encoding="utf-8"))   #加载字符表
    model = build_model(vocab, char_dim, sentence_length,)     #建立模型
    model.load_state_dict(torch.load(model_path))            #加载模型
    x = []                                                  #设置模型为测试模式
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])   #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, round(float(result[i])), result[i])) #打印结果

if __name__ == '__main__':
    main()
    test_strings = ["a", "ab", "abc", "abcd", "abcde"]
    predict(test_strings, "model.pt", "vocab.json")