# Import
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
模型一：简单的npl多分类任务（线性层+交叉熵）
模型二：简单的npl多分类任务（RNN+交叉熵）
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现
"""
#数据生成
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz你我他好再见"  #字符集
    vocab = {}
    for index, char in enumerate(chars):
        vocab[char] = index   #每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab

#随机生成一个样本 #从所有字中选取sentence_length个字
#反之为负样本

def to_one_hot_vec(target,cate_cnt):
    one_hot_target = [0]*cate_cnt
    one_hot_target[target] = 1
    return one_hot_target


def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #指定哪些字出现时为正样本
    contains_english = bool(set("abc") & set(x))
    contains_chinese = bool(set("你我他") & set(x))

    if contains_english and contains_chinese:
        y = 2 #同时包含a/b/c 和 你/我/他
    elif contains_english:
        y = 0 #只包含a/b/c
    elif contains_chinese:
        y = 1 #只包含你/我/他
    else:
        y = 3 #不包含a/b/c/你/我/他
    x = [vocab.get(word, vocab['unk']) for word in x]
    y = to_one_hot_vec(y,4)
    return x,y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)
#建立模型
class TorchModel_linear(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel_linear, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层
        self.pool = nn.AvgPool1d(sentence_length)   #池化层
        self.classify = nn.Linear(vector_dim, 4)     #线性层 out_feature = 分类个数
        self.loss = nn.CrossEntropyLoss()  #loss函数采用交叉熵 cross entropy包含了softmax
        self.activation = nn.functional.softmax

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = self.pool(x.transpose(1, 2)).squeeze() #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim)
        y_pred = self.classify(x)                       #(batch_size, vector_dim) -> (batch_size, 4)
        #y_pred = self.activation(x,dim = 1)                #(batch_size, 4) -> (batch_size, 4)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return self.activation(y_pred,dim = 1)               #输出预测结果

class TorchModel_rnn(nn.Module):
    def __init__(self, vector_dim,hidden_size,sentence_length,vocab):
        super(TorchModel_rnn, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层
        self.rnn = nn.RNN(vector_dim,hidden_size,batch_first=True)
        self.pool = nn.AvgPool1d(sentence_length)   #池化层
        self.classify = nn.Linear(hidden_size, 4)     #线性层 out_feature = 分类个数
        self.loss = nn.CrossEntropyLoss()  #loss函数采用交叉熵 cross entropy包含了softmax
        self.activation = nn.functional.softmax

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x,h = self.rnn(x)                       #(batch_size, sen_len, hidden_size)
        #x = torch.squeeze(h) #取最后一个时间步作为特征表示 or pooling
        x = self.pool(x.transpose(1, 2)).squeeze() #(batch_size, sen_len, hidden) -> (batch_size, hidden)
        y_pred = self.classify(x)                       #(batch_size, vector_dim) -> (batch_size, 4)
        #y_pred = self.activation(x,dim = 1)                #(batch_size, 4) -> (batch_size, 4)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return self.activation(y_pred,dim = 1)               #输出预测结果


#建立模型
def build_model(vocab,char_dim,sentence_length,hidden_size,model_num = 1):
    if model_num == 1:
        model = TorchModel_linear(char_dim, sentence_length, vocab)
    else:
        model = TorchModel_rnn(char_dim,hidden_size,sentence_length,vocab)
    return model

#验证代码
#用来验证每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于验证的样本
    print("本次预测集中样本的属于类别分布是",torch.sum(y,dim=0))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if torch.argmax(y_p).item() == torch.argmax(y_t).item():
                correct +=1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 30        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 1000    #每轮训练总共训练的样本总数
    char_dim =5        #embedding层的维度
    sentence_length = 10   #样本文本长度
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length, hidden_size = 5, model_num=1)  # 1 - linear layer #2 - rnn layer
    #model = build_model(vocab, char_dim, sentence_length,hidden_size = 5,model_num=2) #1 - linear layer #2 - rnn layer
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
    plt.xlabel('epoch')
    plt.ylabel('value')
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
    char_dim = 5  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    #vocab = build_vocab() #加载字符表
    model = build_model(vocab, char_dim, sentence_length, hidden_size=5, model_num=1)  # 建立模型1
    #model = build_model(vocab, char_dim, sentence_length,hidden_size = 5,model_num=2)     #建立模型2
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char,vocab['unk']) for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
        out = torch.argmax(result, dim=1)
    for i, input_string in enumerate(input_strings):
        print('输入:',input_string,'预测类别：', out[i].numpy(),'概率：', result[i].numpy())#打印结果


if __name__ == "__main__":
    main()
    test_strings = ["ffvfeeqqaa", "wwsdfg我oob", "r你qwdyghhz", "nyzkwww还有h"]
    predict("model.pth", "vocab.json", test_strings)
