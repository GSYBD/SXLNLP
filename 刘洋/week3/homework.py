"""
构建一个 用RNN实现的多分类任务
分类详情 当一个字符串中出现“你”则表示为样本1,当出现“我”则表示为样本2,当出现“他”则表示为样本3,当以上都不出现则表示为样本0
"""
import random
import numpy as np
import torch
from torch import nn
import torch.utils.data as Data
import json

#构建模型
class TorchModel(nn.Module):
    def __init__(self, sentence_length, vocab, hidden_size):
        super(TorchModel, self).__init__()
        self.emb = nn.Embedding(len(vocab)+1, hidden_size, padding_idx=0)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.pool = nn.MaxPool1d(sentence_length)
        self.fc = nn.Linear(hidden_size, 4)
        self.loss = nn.functional.cross_entropy

#构建计算过程
    def forward(self, x, y=None):
        x = self.emb(x)
        x = self.rnn(x)[0]
        x = self.pool(x.transpose(1, 2)).squeeze()
        y_pred = self.fc(x)
        if y is not None:
            return self.loss(y_pred, y.view(-1))
        else:
            return y_pred

#构建数据集
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz你我他"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

#构建样本
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 当你出现则表示为样本1
    if set("你") & set(x):
        y = 1
   # 当我出现则表示为样本2
    elif set("我") & set(x):
        y = 2
   # 当他出现则表示为样本3
    elif set("他") & set(x):
        y = 3
   # 当以上都不出现则表示为样本0
    else:
        y= 0
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y

#构建数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


#创建模型
def build_model(sentence_length, vocab, hidden_size):
    model = TorchModel(sentence_length, vocab, hidden_size)
    return model


#预测集
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    # print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 200 - sum(y)))
    correct,wrong = 0,0
    with torch.no_grad():
        y_pred = model(x)
        # print('我是预测值%d'%y_pred)#模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
        print(f'正确的个数为{correct}个，正确率为{correct / (correct + wrong)}')
        return correct / (correct + wrong)



#训练过程
def main():
    batch_size = 20
    lr = 0.002
    train_simple = 5000
    hidden_size = 64
    vocab = build_vocab()
    epoch_size = 10
    sentence_length = 10
    model = build_model(sentence_length, vocab, hidden_size)
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    #加载训练数据
    X, Y = build_dataset(train_simple, vocab, sentence_length)
    dataset = Data.TensorDataset(X, Y)
    data_item = Data.DataLoader(dataset, batch_size, shuffle=True)
    #训练过程
    for epoch in range(epoch_size):
        epoch_loss = []
        model.train()
        for x, y_true in data_item:
            loss = model(x, y_true)
            loss.backward()
            optim.step()
            optim.zero_grad()
            epoch_loss.append(loss.item())
        print("第%d轮 loss = %f" % (epoch + 1, np.mean(epoch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
    #保存模型
    torch.save(model.state_dict(), "model_work3.pt")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#测试集
def predict(model_path, vocab_path, input_strings):
    hidden_size = 64  # 每个字的维度
    sentence_length=10
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(sentence_length, vocab, hidden_size)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
        print(result)
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d"%(input_string, round(float(torch.argmax(result[i])))))  #打印结果


if __name__ == '__main__':
    main()
    test_string=["我abcdabcdi","你abcdabcdi","oabcdabcdi","他abcdabcdi","aabcdabcdi"]
    predict("model_work3.pt", "vocab.json", test_string)