import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
import json


'''
基于RNN的多分类模型实现
'''


class MultiClassRNNModel(nn.Module):
    def __init__(self, vector_dim, vocab, hidden_size):
        super(MultiClassRNNModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.rnn = nn.RNN(vector_dim, hidden_size, bias=False, batch_first=True)
        self.classify = nn.Linear(hidden_size, 6)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)
        rnn_out, hidden = self.rnn(x)
        hidden = hidden.squeeze(0)
        y_pred = self.classify(hidden)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz您"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    remaining_chars = "abcdefghijklmnopqrstuvwxyz"
    x = [random.choice(remaining_chars) for _ in range(sentence_length-1)]
    x.append("您")
    random.shuffle(x)
    chars_index = x.index("您")
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, chars_index


def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def build_model(vocab, char_dim, hidden_size):
    model = MultiClassRNNModel(char_dim,  vocab, hidden_size)
    return model


def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print(f"正确预测个数：{correct}, 正确率：{correct/(correct+wrong)}")
    return correct/(correct+wrong)


def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 500
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.005
    hidden_size = 7
    #  建立字表
    vocab = build_vocab()
    #  建立模型
    model = build_model(vocab, char_dim, hidden_size)
    #  选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    #  训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()    # 梯度归零
            loss = model(x, y)   # 计算loss
            loss.backward()      # 计算梯度
            optim.step()         # 更新权重
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss)}")
        acc = evaluate(model, vocab, sentence_length)   # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    #  保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20    # 每个字的维度
    hidden_size = 7  # 隐藏层维度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, hidden_size)     # 建立模型
    model.load_state_dict(torch.load(model_path))             # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  # 将输入序列化
    model.eval()   # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for vec, res in zip(input_strings, result):
        print(f"输入：{vec}, 预测类别：{torch.argmax(res)}, 概率值：{res}")  # 打印结果


if __name__ == "__main__":
    main()
    test_strings = ["fnv您ee", "您zsdfg", "rqwde您", "nakwa您"]
    predict("model.pth", "vocab.json", test_strings)
