import os
os.environ["KMP_DUPLICTE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

'''
定义模型
使用RNN模型实现多分类任务
'''
class TorchModel(nn.Module):
    def __init__(self, vector_dim, hidden_size, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        self.rnn = nn.RNN(vector_dim, hidden_size, bias=False, batch_first=True)
        self.classify = nn.Linear(hidden_size, sentence_length + 1)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)
        output, hidden = self.rnn(x)
        # print(hidden, "hidden")
        x = hidden.squeeze()
        # print(x, "after squeeze")
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred
   

'''
建立数据集
输入需要的样本数量，需要多少生成多少
'''
def build_vocab():
    chars = "abcdefghijk"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    if 'a' in x:
        y = x.index('a')
    else:
        y = sentence_length

    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


'''
建立模型
'''
def build_model(vocab, char_dim, hidden_size, sentence_length):
    model = TorchModel(char_dim, hidden_size, sentence_length, vocab)
    return model


# 测试代码 用来测试每轮模型的准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    test_sample_num = 200
    x, y = build_dataset(test_sample_num, vocab, sentence_length)
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20
    batch_size = 20
    train_sample = 1000
    char_dim = 10
    hidden_size = 10
    sentence_length = 6
    learning_rate = 0.001

    # 建立字表
    vocab = build_vocab()
    # print(vocab)
    # 建立模型
    model = build_model(vocab, char_dim, hidden_size, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample // batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            # print(x, "build dataset x")
            # print(y, "build dataset y")
            optim.zero_grad()   # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()     # 计算梯度
            optim.step()        # 梯度/权重更新
            watch_loss.append(loss.item())
        print("===============\n第%d轮  平均loss：%f" % (epoch + 1, np.mean(watch_loss)))

        acc = evaluate(model, vocab, sentence_length)   # 测试本轮模式结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


def predict(model_path, vocab_path, input_strings):
    char_dim = 10
    hidden_size = 10
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, hidden_size, sentence_length)
    model.load_state_dict(torch.load(model_path))

    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, torch.argmax(result[i]), result[i]))

if __name__ == "__main__":
    # main()

    test_string = ["kijabc", "gijkbc", "gkijad", "kijhde"]
    predict("model.pth", "vocab.json", test_string)