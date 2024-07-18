import json
import random
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
"""

使用rnn做文本分类
6个字的文本，'abcdef',第一次出现a 的位置为类别，

"""


# rnn模型
class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, hidden_size, vocab):
        super(TorchModel, self).__init__()

        self.embedding = nn.Embedding(len(vocab), vector_dim)  # 创建embedding层 20wei
        # rnn层，输入20维，隐藏层20维，输出hidden_size维
        self.layer = nn.RNN(vector_dim, hidden_size, bias=False, batch_first=True)

        self.classify = nn.Linear(hidden_size, sentence_length)
        self.loss = nn.CrossEntropyLoss()  # 定义交叉熵

    def forward(self, x, y=None):
        x = self.embedding(x)  # 向量化词 20 * 6 * 20
        # rnn层 output 20 * 6 * 30
        # rnn层 hidden [1, 20, 30] 变成 20 * 30 只要output的最后一维
        output, x = self.layer(x)
        x = x.squeeze()  # 20 * 30 相当于 20个样本 每个样本变成 1 * 30
        y_pred = self.classify(x)  # 线性层 转为 20 * 6

        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred  # 输出预测结果 6维


def build_model(vocab, sententce_lenth, char_dim, hidden_size):
    return TorchModel(char_dim, sententce_lenth, hidden_size, vocab)


# 字符集
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 26
    return vocab


# 随机生成一个样本
# a第一次出现的位置为类别

def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length - 1)]
    # 生成一个随机索引
    index = random.randint(0, len(x))
    # 在该索引位置插入值，为了保证有a
    # TODO 没想明白没有a的情况要怎么表示
    obj = 'a'
    x.insert(index, obj)
    y = x.index(obj)

    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，为了做embedding
    return x, y


# 创建训练数据
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)
    print("本次预测集中共有%d个样本" % (sum(y)))

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20
    batch_size = 20
    train_sample = 500
    char_dim = 20
    sentence_length = 6
    hidden_size = 30
    learning_rate = 0.005
    vocab = build_vocab()

    # 建模型
    """
    训练参数
    embedding.weight 28*20
    layer.weight_ih_l0 30 * 20 input * hidden
    layer.weight_hh_l0 30, 30  hidden * hidden
    """
    model = build_model(vocab, sentence_length, char_dim, hidden_size)

    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            # 构建训练数据
            x, y = build_dataset(batch_size, vocab, sentence_length)
            # 开始训练
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
            optim.zero_grad()  # 梯度归零

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), 'model.pth')
    print('state_dict2', model.state_dict())

    # 保存词表
    writer = open('vocab.json', 'w', encoding='utf8')
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    hidden_size = 30
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表

    model = build_model(vocab, sentence_length, char_dim, hidden_size)
    model.load_state_dict(torch.load(model_path))
    print(vocab)
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%s" % (input_string, torch.argmax(result[i]).item(), result[i]))  # 打印结果


if __name__ == "__main__":
    # main()
    test_strings = ["favfee", "wzsdfa", "aqwdeg", "nakwaw"]
    predict("model.pth", "vocab.json", test_strings)
