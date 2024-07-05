import random
import numpy as np
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt

#定义RNN模型
class TorchModel(nn.Module):
    def __init__(self, vocab, embedding_dim):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_size=5, bias=False, batch_first=True)
        self.fc = nn.Linear(5, len(vocab))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embedding(x)
        rnn_out, _ = self.rnn(x)
        rnn_out = rnn_out[:, -1, :]
        y_pred = self.fc(rnn_out)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

#创建字符表
def build_vocab():
    vocab = {'pad': 0}
    chars = '你好avu'
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab
# print(build_vocab())
vocab = build_vocab()

#创建样本
def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length - 1)]
    ni_index = random.randint(0, sentence_length - 1)
    x.insert(ni_index, '你')
    # print(x)
    for i, char in enumerate(x):
        if char == "你" and i == 0:
            y = 0
        elif char == "你" and i == 1:
            y = 1
        elif char == "你" and i == 2:
            y = 2
        elif char == "你" and i == 3:
            y = 3
        elif char == "你" and i == 4:
            y = 4
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

#创建数据集
def build_samples(sample_num, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_num):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)
# print(build_samples(4, vocab, 5))

def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_samples(200, vocab, 5)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print(f'正确的个数为{correct}个，正确率为{correct/(correct + wrong)}')
    return correct/(correct + wrong)

#创建模型
def build_model(vocab, embedding_dim):
    model = TorchModel(vocab, embedding_dim)
    return model

#训练模型
def main():
    #参数初始化
    embedding_dim = 20
    batch_size = 20
    lr = 0.005
    sample_num = 1000
    sentence_length = 5
    epoch_num = 20

    vocab = build_vocab()
    model = build_model(vocab, embedding_dim)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(sample_num / batch_size)):
            x, y = build_samples(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
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
# main()

def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
        print(result)
    for i, input_string in enumerate(input_strings):
        # print(i,input_string)
        print(result[i])
        print("输入：%s, 预测类别：%d"%(input_string, round(float(torch.argmax(result[i])))))  #打印结果

test_strings = ["你a好uv", "你好a你v", "a好av你"]
predict("model.pth", "vocab.json", test_strings)