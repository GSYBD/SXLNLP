import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import json
import seaborn as sns


# 构建模型类
class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim,padding_idx=0)
        self.rnn = nn.RNN(vector_dim,hidden_size=20,dropout=0.1,num_layers=2,nonlinearity="relu",batch_first=True,bias=False)
        self.classify = nn.Linear(20,  1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)
        x,h = self.rnn(x)
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def  build_vocab():
    vocab = {"pad": 0}
    chars = "abcdefghijklmnopqrstuvwxyz"
    for idx, char in enumerate(chars):
        vocab[char] = idx + 1
    vocab['我'] = len(vocab)
    vocab['你'] = len(vocab)
    vocab['他'] = len(vocab)
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]  # 按次数获取随机键
    if set('a') & set(x):
        y = x.index("a")
    else:
        y = 0
    x = [vocab.get(word, "unk") for word in x]  # 根据x中的键 获取值
    return x, y

def build_dataset(sample_length, sentence_length, vocab):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, sentence_length, vocab)
    print('本轮测试样本中正样本%d个，负样本%d个' % (sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if y_p.argmax() == y_t:
                correct += 1
            else:
                wrong += 1
    print('本轮正确样本个数：%d个，正确率为：%f' % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 训练开始

def main():
    epoch_num = 20
    batch_size = 10
    train_sample = 500
    learning_rate = 0.005  # 调整学习率 控制收敛
    hidden_size = 20
    sentence_length = 6
    char_dim = 20
    vocab = build_vocab()

    model = TorchModel(char_dim, sentence_length,vocab)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, sentence_length, vocab)
            optim.zero_grad()  # 如果关闭梯度清理，训练会一直波动
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print('---------第%d轮训练,平均loss；%f' % ((epoch + 1), np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])

    # 绘图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model_rnn")
    # 保存词典
    writer = open("vocab", "w", encoding="utf8")  # 创建词典
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))  # 写入文件
    writer.close()  # 关闭文件
    return


def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    model_path = "model_rnn"
    vocab_path = "vocab.json"
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = TorchModel(char_dim, sentence_length, vocab)
    model.load_state_dict(torch.load((model_path)))
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])
    model.eval()
    with torch.no_grad():
        res = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print("输入 %s,预测类别 %d，概率值%f" % (input_string,round(float(res[i])),res[i])) #round是将预测值四舍五入了


if __name__ == "__main__":
    main()
    test_strings = ["fnbaee", "wzsyfg", "rqwteg", "nakwww"]
    predict("model.pth", "vocab.json", test_strings)
