import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from torch import optim
"""
构建一个nlp任务，不用pooling层，用rnn做一个多分类，结合交叉熵
特定字符，“你"保证一定出现在字符串，“你”在文本里面第几位，就属于第几类
"""
def build_vocab(): # vocab词汇
    chars = "你我他好坏上中下asdfghjk" # 字符集
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab) + 1
    return vocab
# # 生成字典，测试
# vocab = build_vocab()
# print(vocab)

# 位置
def find_indices(lst, value):  # 判断某个字符在文本里面的索引位置
    return [index for index, item in enumerate(lst) if item == value]

def build_sample(vocab, sentence_length): # sample样本

    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)] # 将样本转换成字典里面对应的value
    #print(x) # 一个文本
    if x.count("你") == 1:
        indices = find_indices(x, "你")
        y = [idx for idx in indices]
    else:
        return build_sample(vocab, sentence_length) # 不满足条件，重新调用样本
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将输入序列化,获取索引的位置
    # print(x)  # 例子：[2, 4, 9, 7, 4, 7, 0, 0, 1, 0]
    # print(y)  # 例子：[9]
    return x, y
def build_dataset(vocab, sample_length, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(np.array(dataset_y).flatten())
vocab = build_vocab()
sentence_length = len(vocab) - 1  # 几列(有多少个字符)
sample_num = 3  # 几行（几个样本）
dataset_x, dataset_y = build_dataset(vocab,sample_num,sentence_length) # sample_num几行，  sentence_length 几列   sample_num（几行文本)，sentence_length几列(文本长度)sentence句子长度
print("dataset_x",dataset_x)
print("dataset_y",dataset_y)  # 一维  sample_num列
vocab = build_vocab()
x,y = build_dataset(vocab,3,5)
print("x",x)
print("y",y)

# 3.创建模型
class TorchModel(nn.Module):
    def __init__(self, vocab, char_dim, sentence_length, hidden_size,num_class):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, char_dim, padding_idx=0)   # Embedding层
        self.nb = nn.LayerNorm(char_dim)          # 规划层
        self.dropout = nn.Dropout(0.3)            # dropout
        self.rnn_layer = nn.LSTM(input_size=char_dim, hidden_size=hidden_size, batch_first=True)  # RNN层
        self.classify = nn.Linear(hidden_size,num_class)     # 线性层
        self.loss = nn.CrossEntropyLoss()                   # 交叉熵

    def forward(self, x, y=None):
        x = self.embedding(x)                 # （batch_size,sen_len） ----- （batch_size,sen_len,char_dim）
        x = self.nb(x)                        # 规划层
        x = self.dropout(x)
        x, (ht,ct) = self.rnn_layer(x) # （batch_size,sen_len,char_dim） ----- （batch_size,sen_len,hidden_size） # 初始隐藏状态张量 ht,初始细胞状态张量 ct0
        y_pred = self.classify(ht.squeeze()) # （batch_size,sen_len,hidden_size ---- batch_size,sen_len,num_class）
        # print("y_pred",y_pred)
        if y is not None:
            return self.loss(y_pred, y)  # 计算损失值
        else:
            return y_pred
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(vocab,5000 , sample_length)  # 建立200个用于测试的样本  x训练集 y是真实值
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测   预测值y_pred
        y_pred = torch.argmax(y_pred, dim=1)  # dim维 # 返回值大的索引
        correct = (y_pred == y).sum().item()  # .sum()：这是对布尔类型的张量进行求和操作。在 Python 中，True 被解释为 1，False 被解释为 0， .item()：这是将张量中的单个元素提取为 Python 标量的方法。在这种情况下，由于我们只有一个数值结果，所以可以使用 .item() 将其提取为 Python 中的普通整数。
        wrong = len(y) - correct
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num = 10         # 训练轮数
    batch_size = 20        # 每次训练样本个数
    char_dim = 50          # 每轮训练总共训练的样本总数
    hidden_size = 100       # 每个字的维度
    vocab = build_vocab()
    sentence_length = len(vocab) - 1   # 样本文本长度
    sample_num = 5000               # 样本数量
    num_class = sentence_length
    log = []
    model = TorchModel(vocab, char_dim, sentence_length, hidden_size,num_class) # 创建模型
    optimizer = optim.Adam(model.parameters(), lr=1e-3)   # 选择优化器
    dataset_x, dataset_y = build_dataset(vocab,sample_num,sentence_length) # 创建训练数据集
    for epoch in range(epoch_num):
        model.train()  # 训练模式
        watch_loss = []
        for batch_index in range(sample_num // batch_size):
            x = dataset_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = dataset_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y) # 计算loss
            loss.backward()   # 计算梯度
            optimizer.step() # #  更新权重
            optimizer.zero_grad()  # 梯度置0
            watch_loss.append(loss.item())
        print("第%d轮,loss=%f" % (epoch + 1,np.mean(watch_loss)))  # np.mean()求平均
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    # 保存模型
    torch.save(model.state_dict(), 'model.pth')
    print(log)
# 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
def predict(model_path,input_str):  # model_path训练好模型文本，input_str文本
    vocab = build_vocab()
    char_dim = 50
    hidden_size = 100
    sentence_length = len(vocab) - 1
    num_class = sentence_length
    model = TorchModel(vocab, char_dim, sentence_length, hidden_size,num_class)
    model.load_state_dict(torch.load(model_path))   # 加载训练好的权重
    model.eval()  # 预测模式
    X = []
    for sentence in input_str:
        x = [vocab.get(char) for char in sentence] # # 只需要字符的索引就行了，因为预测的结果为索引位置，对文本遍历，列表推导式和字典的 get() 方法来将文本中的每个字符转换为对应的词汇表中的索引
        X.append(x)
    X = torch.LongTensor(X)
    with torch.no_grad():
        y_pred = model(X)  # 预测结果
    for y_p, y_t in zip(y_pred, input_str):  # zip函数将多个可迭代对象作为参数，并返回一个由对应位置的元素组成的元组的迭代器
        i = y_t.index('你')
        print("正确位置:%d,预测位置:%d,是否正确:%s" % (i, torch.argmax(y_p), (torch.argmax(y_p) == i)))

if __name__ == '__main__':
    main()
    print("模型创建成功！")
    input_str = ['坏上中下你我他好asdfghjk',
                 '上中下asd你我他好坏fghjk',
                 'asdf你我他好坏上中下ghjk',
                 'hjk你我他好坏上中下asdfg',
                 'as你我他好坏上中下dfghjk',
                 'gh好坏你我他上中下asdfjk']

    predict('model.pth', input_str)
