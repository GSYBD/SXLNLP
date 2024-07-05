import torch
import torch.nn as nn
import numpy as np
import random
import json
from torch.utils.data import TensorDataset, DataLoader
""""
基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中某特定字符，如p、o、q分别出现在第几个位置，就属于第几类。

"""
class TorchModel(nn.Module):
    def __init__(self, vector_dim, hidden_size, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)
        self.rnn = nn.RNN(vector_dim, hidden_size, bias=False, batch_first=True)
        self.pool = nn.AvgPool1d(sentence_length)
        self.classify = nn.Linear(hidden_size, 3)  # 假设类别数为3
        self.softmax = torch.nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze()
        x = self.classify(x)
        y_pred = self.softmax(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab

# 随机生成一个样本
# 从所有字中选取sentence_length个字
def build_sample(vocab, sentence_length):
    vocab_keys = list(vocab.keys())
    x = [random.choice(vocab_keys) for _ in range(sentence_length)]
    positions = {'p': (0, 1), 'o': (2, 3), 'q': (4, 5)}
    positions_in_text = {char: (None, None) for char in positions.keys()}


    for i, char in enumerate(x):
        if char in positions_in_text:
            positions_in_text[char] = (i, char)
            if len(positions_in_text) == 3:
                break

    if positions_in_text:
        category_num = list(positions_in_text.values()).index(positions_in_text["p"][1]) + 1
        return positions_in_text["p"][1], positions_in_text
    else:
        return x, 0

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim, sentence_length):
    hidden_size = 4
    model = TorchModel(char_dim, hidden_size, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if torch.argmax(y_p) == torch.argmax(y_t):
                correct += 1
            else:
                wrong += 1
    print(f"正确预测个数：{correct}, 正确率：{correct / (correct + wrong)}")
    return correct / (correct + wrong)

def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 500
    char_dim = 20
    sentence_length = 5
    learning_rate = 0.005
    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss：{np.mean(watch_loss)}")
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    torch.save(model.state_dict(), "model.pth")
    with open("vocab.json", "w", encoding="utf8") as writer:
        writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    x = [[vocab[char] for char in input_string] for input_string in input_strings]
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))
    for i, input_string in enumerate(input_strings):
        print(f"输入：{input_string}, 预测类别：{int(round(float(result[i])))}, 概率值：{result[i]}")

if __name__ == "__main__":
    main()
    test_strings = ["fnufee", "wzsdfg", "rqwdeg", "nakuwww"]
    predict("model.pth", "vocab.json", test_strings)