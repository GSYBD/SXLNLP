
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch使用RNN进行文本的多分类任务
对于对个文本,每个文本中一定有a  该文本的类别是'a'第一次的位置下标

"""

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

# 生成随机样本
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 获取随机下标 替换为'a'
    random_index = random.randint(0, len(x) - 1)
    x[random_index] = 'a'
    # 根据第一个'a'出现的位置判断类别,即下标
    y = x.index('a')
    # 将字转换成序号，为了做embedding
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 模型的设置
class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)   # embedding层
        self.rnn = nn.RNN(vector_dim, sentence_length, bias=False, batch_first=True)  # RNN
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        y_pred = x
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#用来测试每轮模型的准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)   #建立200个用于测试的样本
    print("本次预测集中共有200个样本")
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        y_pred = torch.argmax(y_pred, dim=1)  # 选择最大值的索引作为预测值
        correct = (y_pred == y).sum().item()  # 计算正确的预测数量
        wrong = len(y) - correct
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 100        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 17         #每个字的维度
    sentence_length = 10   #样本文本长度
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
    torch.save(model.state_dict(), "cllThirdModel.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

# 预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 17  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model(torch.LongTensor(x))  # 模型预测
        result = torch.argmax(result, dim=1)  # 选择最大值的索引作为预测值
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d" % (input_string, result[i].item())) #打印结果

if __name__ == "__main__":
    # main()
    test_strings = ["fnvfeeayhf", "wzsadfgaef", "raqwdwdgeg", "aadrnakwww"]
    predict("cllThirdModel.pth", "vocab.json", test_strings)