#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中字符“您”出现的位置

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim,  vocab,hidden_size):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim,padding_idx=0)  #embedding层
        self.rnn=nn.RNN(vector_dim, hidden_size, bias=False, batch_first=True)    #RNN,hidden_size=6
        self.classify = nn.Linear(hidden_size, 6)     #线性层
        self.loss = nn.functional.cross_entropy #loss函数采用交叉熵

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        rnn_out, hidden = self.rnn(x)              #rnn_out:(batch_size,sentence_length,hidden_size);hidden:(1,batch_size,hidden_size)
        hidden=hidden.squeeze(0)                   #(1,batch_size,hidden_size) -> (batch_size, hidden_size)
        y_pred=self.classify(hidden)               #(batch_size,6)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz您"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #28
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字

def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length-1个字，可能重复
    remaining_chars="abcdefghijklmnopqrstuvwxyz"
    x = [random.choice(remaining_chars) for _ in range(sentence_length-1)]
    x.append("您")
    random.shuffle(x)#随机打乱
    chars_index=x.index("您")
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding，vocab.get(word)：尝试从词汇表中获取单词 word 对应的索引，如果单词 word 不在词汇表中，get 方法将返回 vocab['unk'] 的值
    return x, chars_index

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab, char_dim,hidden_size):
    model = TorchModel(char_dim,  vocab, hidden_size)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 20        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500   #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率,任务比较简单，这里学习率放大一些，加快训练
    hidden_size = 7      #隐藏层维度
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim,hidden_size)
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
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    hidden_size = 7  # 隐藏层维度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim,hidden_size)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for vec,res in zip(input_strings,result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res)) #打印结果

if __name__ == "__main__":
    main()
    test_strings = ["fnv您ee", "您zsdfg", "rqwde您", "nakwa您"]
    predict("model.pth", "vocab.json", test_strings)
