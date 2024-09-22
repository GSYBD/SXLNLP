# coding:utf8
import random
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
输入一个字符串，根据字符“优”所在位置进行分类
对比rnn和pooling做法
"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  # embedding层
        # 可以自行尝试切换使用rnn或pooling
        # self.pool = nn.AvgPool1d(sentence_length)   # 池化层
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        # +1的原因是可能出现“优”不存在的情况，那时的真实label在构造数据时设为了sentence_length
        self.classify = nn.Linear(vector_dim, sentence_length + 1)     
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)           
        # 使用pooling的情况
        # x = x.transpose(1, 2)           
        # x = self.pool(x)                
        # x = x.squeeze()   
        # 使用rnn的情况              
        rnn_out, hidden = self.rnn(x)
        x = rnn_out[:, -1, :]  # 或者写hidden.squeeze()也是可以的，因为rnn的hidden就是最后一个位置的输出

        # 接线性层做分类
        y_pred = self.classify(x)            
        if y is not None:
            return self.loss(y_pred, y)   # 预测值和真实值计算损失
        else:
            return y_pred                 # 输出预测结果

# 字符集随便挑了一些汉字，实际上还可以扩充
# 为每个字生成一个标号
# {"优":1, "良":2, "中":3...}
# 优良中 -> [1,2,3]
def build_vocab():
    chars = "优良中差及格不及格"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1   # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 未知字符的标号
    return vocab

# 随机生成一个样本
def build_sample(vocab, sentence_length):
    chars = list(vocab.keys())
    chars.remove("pad")
    chars.remove("unk")
    if sentence_length > len(chars):
        sentence_length = len(chars)
    x = random.sample(chars, sentence_length)
    # 指定哪些字出现时为正样本
    if "优" in x:
        y = x.index("优")
    else:
        y = sentence_length
    x = [vocab.get(word, vocab['unk']) for word in x]   # 将字转换成序号，为了做embedding
    return x, y

# 建立数据集
# 输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   # 建立200个用于测试的样本
    print("本次预测集中共有%d个样本" % (len(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 20        # 训练轮数
    batch_size = 40       # 每次训练样本个数
    train_sample = 1000    # 每轮训练总共训练的样本总数
    char_dim = 30         # 每个字的维度
    sentence_length = 7   # 样本文本长度，调整为小于字符集长度
    learning_rate = 0.001 # 学习率
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
            x, y = build_dataset(batch_size, vocab, sentence_length) # 构造一组训练样本
            optim.zero_grad()    # 梯度归零
            loss = model(x, y)   # 计算loss
            loss.backward()      # 计算梯度
            optim.step()         # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 30  # 每个字的维度
    sentence_length = 7  # 样本文本长度，调整为小于字符集长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)     # 建立模型
    model.load_state_dict(torch.load(model_path))             # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  # 将输入序列化
    model.eval()   # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, torch.argmax(result[i]), result[i])) # 打印结果

if __name__ == "__main__":
    main()
    test_strings = ["及格不优良中差格不", "中良不优及差格格", "差良及优格不中", "优及良不及格不"]
    predict("model.pth", "vocab.json", test_strings)
