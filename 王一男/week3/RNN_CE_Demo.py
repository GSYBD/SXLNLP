# coding:utf8

import torch
from torch import nn, optim
import numpy as np

import random

"""

第三周作业
使用RNN和交叉熵实现一个文本的多分类任务
规律：使用RNN和交叉熵做一个多分类任务，找一个字符串，这个字符串在第几个坐标就属于第几类。

"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sequence_len, vocab, hidden_size):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=vector_dim)
        self.rnn = nn.RNN(input_size=vector_dim, hidden_size=hidden_size, batch_first=True)
        self.classify = nn.Linear(in_features=hidden_size, out_features=sequence_len)
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)              # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x, ht = self.rnn(x)
        # 取最后一个隐藏状态的话，(batch_size, sen_len, vector_dim) -> (batch_size, 1, hidden_size)
        y_pred = self.classify(ht.squeeze())  # squeeze去除sen_len维, 变为class_num维特征向量
        if y is not None:
            return self.loss(input=y_pred, target=y)   # 预测值和真实值计算损失
        else:
            return y_pred     # 输出预测结果


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyzA"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1   # 每个字对应一个序号
    vocab['unk'] = 99  # 26
    return vocab


def build_dataset(vocab, sample_num, sequence_len):
    X = []
    Y = []
    for _ in range(sample_num):
        x = list(vocab.keys())
        x.remove('unk')
        x.remove('pad')
        x.remove('A')     # 确保数据集里只有一个'A'?
        x = [random.choice(x) for _ in range(sequence_len-1)]
        x.append('A')     # 追加字符并打乱顺序，确保存在该字符
        random.shuffle(x)
        y = x.index('A')  # 需要查找的字符
        x = [vocab.get(char, vocab['unk']) for char in x]
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)


def main():
    # 配置参数
    epoch_num = 20        # 训练轮数
    batch_size = 20       # 每次训练样本个数
    train_sample = 5000   # 每轮训练总共训练的样本总数
    char_dim = 6          # 每个字的维度
    sentence_length = 12  # 样本文本长度
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = TorchModel(vector_dim=char_dim, sequence_len=sentence_length, vocab=vocab, hidden_size=20)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset_x, dataset_y = build_dataset(vocab=vocab, sample_num=train_sample, sequence_len=sentence_length)
    # log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample // batch_size)):
            x = dataset_x[batch * batch_size: (batch + 1) * batch_size]
            y = dataset_y[batch * batch_size: (batch + 1) * batch_size]
            loss = model(x, y)     # 计算loss
            loss.backward()        # 计算梯度
            optimizer.step()           # 更新权重
            optimizer.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=================\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        # log.append([acc, np.mean(watch_loss)])
    torch.save(model.state_dict(), 'model.pth')
    return


def predict(model_path, sample_num):
    vocab = build_vocab()
    char_dim = 6            # 每个字的维度
    sentence_length = 12    # 样本文本长度
    model = TorchModel(vector_dim=char_dim, sequence_len=sentence_length, vocab=vocab, hidden_size=20)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    X = []
    Y = []
    X, Y = build_dataset(vocab=vocab, sample_num=sample_num, sequence_len=sentence_length)
    with torch.no_grad():
        y_pred = model(X)
    for y_p, y_t in zip(y_pred, Y):
        print("正确位置:%d,预测位置%d,是否正确:%s" % (y_t, int(torch.argmax(y_p)), str(y_t == torch.argmax(y_p))))


if __name__ == "__main__":
    main()
    predict('model.pth', 20)
