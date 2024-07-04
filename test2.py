import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, sentence_length, vocab):
        super(TorchRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab)+2, input_size, padding_idx=0)
        self.pool = nn.AvgPool1d(sentence_length)
        self.layer = nn.RNN(input_size, hidden_size, bias=False, batch_first=True)
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss

    def forward(self, x, y=None):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze()
        x = self.layer(x)
        # print(x)
        # x = torch.stack(x)
        x = x[0]
        # print(x)
        # print(type(x))
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

# 字符集
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index+1
    vocab['unk'] = len(vocab)
    vocab["你好"] = len(vocab) + 1
    vocab["世界"] = len(vocab) + 2
    return vocab

# 随机生成样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for label in range(sentence_length)]
    # if set("abc") & set(x):
    if '你好' in x or '世界' in x:
        y = 1
    else:
        y = 0
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

#测试每轮模型准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1  # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    #配置参数
    epoch_num = 20        #训练轮数
    batch_size = 10       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    input_size = 10
    hidden_size = 1
    sentence_length = 10   #样本文本长度
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = TorchRNN(input_size, hidden_size, sentence_length, vocab)
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

if __name__ == "__main__":
    main()




# vocab = build_vocab()
# x, y = build_dataset(10, vocab, 10)
# model = TorchRNN(10, 1, 10, vocab)
# cll = evaluate(model,vocab,10)
# print(x)
# print(model.state_dict())
# y_pred = model.forward(x)
# print(y_pred)
# print(y)

