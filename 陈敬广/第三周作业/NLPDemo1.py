import json
import random
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

'''
基于pytorch的网络编写
实现一个网络完成nlp任务
特定字符出现在样本的索引表示分类
'''


# 建立网络模型
class TorchModel(nn.Module):
    # vocab:词表；vector_dim：样本字符映射向量维度；sentence_length：样本字符数量
    def __init__(self, vocab, vector_dim, sentence_length):
        super(TorchModel, self).__init__()
        # embedding层，字符向量化
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        # pooling 词化层
        # self.pool = nn.MaxPool1d(sentence_length)
        # RNN 循环网络层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        self.linear = nn.Linear(vector_dim, sentence_length)
        # batch normal 层
        self.bn = nn.BatchNorm1d(vector_dim)
        # dropout 层
        self.drop = nn.Dropout(0.1)
        # 激活函数
        # self.activate = nn.functional.sigmoid
        # loss 损失函数 交叉熵
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)
        output, h = self.rnn(x)
        # x = x.transpose(1, 2)
        # self.pool(x)
        # x = x.squeeze()
        # h最后一次输出
        x = h.squeeze()  # (1x30x20) --> (30x20)
        # x = self.bn(x)
        # x = self.drop(x)
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

# 建立字符集，词表
def build_vocab():
    chars = 'abcdefghijklmnopqrstuvwxyz'
    vocab = {'pad':0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab

# 根据词表建立一个样本
def build_sample(vocab, sentence_length):
    # 随机从词表中选取sentence_length-1个字，剩余一个是特定字符c
    keyList = list(vocab.keys())
    keyList.remove("c")
    x = [random.choice(keyList) for _ in range(sentence_length-1)]
    random_index = random.randint(0, len(x))
    x.insert(random_index,'c')
    # 见字符转换为数字
    x = [vocab.get(char, vocab['unk']) for char in x]
    return x, random_index

# 建立样本集
def build_dataset(sample_length,vocab, sentence_length):
    X = []
    Y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)

# 建立模型
def build_model(vocab,vector_dim,sentence_length):
    return TorchModel(vocab,vector_dim,sentence_length)

# 评估函数
def evaluate(model,vocab,sentence_length):
    model.eval()
    x, y = build_dataset(100,vocab,sentence_length)
    y_pred = model.forward(x)
    correct, wrong = 0, 0
    with torch.no_grad():
        for y_p,y_t in zip(y_pred, y):
            if torch.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
    print('正确预测个数：%d,正确率：%f' % (correct,correct / (correct + wrong)))
    return correct / (correct + wrong)


# 训练过程
def main():
    epoch_num = 50  # 训练轮数
    batch_size = 30  # 每次训练的样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    vector_dim = 20  # 每个字的向量维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.001  # 学习率

    # 建立词表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab,vector_dim,sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 开始训练
    for i in range(epoch_num):
        model.train()
        watch_loss = []
        for j in range(train_sample // batch_size):
            x, y = build_dataset(batch_size,vocab, sentence_length)
            # 损失值
            loss = model.forward(x,y)
            # 梯度计算，反向传播
            loss.backward()
            # 权重更新
            optim.step()
            # 梯度归零
            optim.zero_grad()
            watch_loss.append(loss.item())
        print('==========\n第%d轮平均loss:%f' % ( i + 1, np.mean(watch_loss)))
        acc = evaluate(model,vocab,sentence_length)
        if acc > 0.99999:
            print('==========\n第%d轮跳出循环' %(i + 1))
            break
        log.append([acc,np.mean(watch_loss)])

    # 画图
    plt.plot(range(len(log)),[l[0] for l in log],label = 'acc')
    plt.plot(range(len(log)),[l[1] for l in log],label = 'loss')
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(),'model.pth')
    # 保存词表
    writer = open('vocab.json','w',encoding='utf-8')
    writer.write(json.dumps(vocab,ensure_ascii=False,indent=2))
    writer.close()


def predict(model_path, vocab_path, input_strings):
    vector_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, 'r', encoding='utf-8'))
    model = build_model(vocab, vector_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    x = []
    # 将输入字符列表转换为序列化数字
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
        print(np.argmax(result.numpy()[1]))
    for i,input_string in enumerate(input_strings):
        print('c.index:%d, result[i].maxp.index: %d' % (input_string.index('c'), np.argmax(result.numpy()[i])))


if __name__ == "__main__":
    # main()
    input_strings = ["abchrj", "abjicj", "abtzhc", "chrhuj", "hcrhuj", "hrhcuj", "chrhuj"]
    predict("model.pth", "vocab.json", input_strings)