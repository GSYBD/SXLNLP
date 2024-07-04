import random
import torch.nn as nn
import torch
import json
import matplotlib.pyplot as plt
import numpy as np


# 任务：根据特定字符在文本中(例如字母‘f’)出现的位置判断文本类型，特定字符出现在第一个位置表示文本类型为0，出现在第二个位置表示文本类型为1，以此类推。


# 定义模型
class Mymodel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(Mymodel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(vector_dim, sentence_length, bias=True, batch_first=True)  # RNN层
        self.ln = torch.nn.LayerNorm(sentence_length)  # 归一化层
        self.dp = nn.Dropout(p=0.3)  # dropout层
        self.pool = nn.AvgPool1d(sentence_length)  # 池化层
        self.activation = torch.softmax  # 激活函数
        self.loss_function = nn.functional.cross_entropy  # 损失函数

    def forward(self, x, y=None):
        x = self.embedding(x)  # 输入x，得到嵌入层输出
        x = self.dp(x)  # 应用dropout层
        x, _ = self.rnn(x)  # 输入嵌入层输出，得到RNN层输出
        x = x.transpose(1, 2)  # (batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        x = self.pool(x)  # (batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        x = x.squeeze()  # (batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        x = self.ln(x)  # 应用归一化层
        y_pred = self.activation(x, dim=1)  # 应用激活函数
        if y is not None:
            # print("y_pred:", y_pred)
            # print("y:", y)
            loss = self.loss_function(y_pred, y)  # 如果y不为空，计算损失
            return loss
        else:
            return y_pred


# 构造单个样本，输出 text, label
def generate_sample(vocab, sentence_length):
    text = ""
    # 随机生成一个lable
    label = random.randint(0, sentence_length - 1)
    for i in range(sentence_length):
        if i == label:
            text += 'f'
        else:
            # 随机选择一个非'f'字符
            char = random.choice(list(vocab.keys()))
            while char == 'f' or char == 'unk' or char == 'pad':
                char = random.choice(list(vocab.keys()))
            text += char

    # 将文本转换为数字
    text_num = [vocab.get(char, vocab['unk']) for char in text]
    # print("文本:", text)
    # print("标签:", label)
    # print("文本转数字", text_num)
    return text_num, label


# 词表
vocab = {'pad': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'unk': 11}


# 构造数据集
def build_dataset(vocab, sentence_length, batch_size):
    x = []
    y = []
    for i in range(batch_size):
        text_num, label = generate_sample(vocab, sentence_length)
        x.append(text_num)
        y.append(label)
    return torch.LongTensor(x), torch.LongTensor(y)


# 建立模型
def build_model(vector_dim, sentence_length, vocab):
    model = Mymodel(vector_dim, sentence_length, vocab)
    return model


# 测试准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(vocab, sentence_length, 200)  # 建立200个用于测试的样本
    correct = 0
    wrong = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == y_t:
                correct += 1  # 判断正确
            else:
                wrong += 1  # 判断错误
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 训练模型
def main():
    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 50  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    vector_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.01  # 学习率

    # 建立模型
    model = build_model(vector_dim, sentence_length, vocab)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(vocab, sentence_length, batch_size)  # 构造一组训练样本
            # print(x, y)
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
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
    vector_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vector_dim, sentence_length, vocab)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, torch.argmax(result[i]).item(), torch.max(result[i])))  # 打印结果


if __name__ == "__main__":
    # main()
    test_strings = ["fbciee", "ccadfg", "jfadeg", "aafdee"]
    predict("model.pth", "vocab.json", test_strings)
