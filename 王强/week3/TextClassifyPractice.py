#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch的网络编写
实现一个RNN网络完成以下任务
给定一个仅带1个'中'的中文字符的字符串(其他都是英文字母)，判断文本中中文字符出现在第几个位置
"""

class TextMulitClassifyModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, num_classes):
        super(TextMulitClassifyModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层
        #todo:这里比较坑，如果经过池化层会把文本顺序信息给丢失(例如abc和bac经过池化处理后都是一样的，但是顺序信息其实丢失了)，导致准确率很低
        # self.pool = nn.AvgPool1d(sentence_length)   #池化层
        self.classify = nn.RNN(vector_dim, num_classes, bias=False, batch_first=True)     #RNN层
        self.dp_layer = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(num_classes)
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉墒

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        # print("x shape is:{}".format(x.shape))
        # x = x.transpose(1, 2)                      #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        # x = self.pool(x)                           #(batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        # x = x.squeeze()                            #(batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        # x = self.dp_layer(x)
        # 这里需要注意RNN层输出的是一个元组(output, hidden)
        output,hidden = self.classify(x)          #(batch_size, sen_len, vector_dim) -> ((batch_size, sen_len, num_classes),(1,batch_size, sen_len))
        # print("output shape is:{}, hidden shape is:{}".format(output.shape, hidden.shape))
        output = output[:, -1, :]
        # print("out put shape is:{}".format(output.shape))
        # y_pred = self.dp_layer(output)
        # y_pred = self.bn(output)
        y_pred = output
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz中"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #27
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length-1个字，然后将'中'字符随机插入到数组中的某个位置
def build_sample(vocab, sentence_length):
    # 先排除词表中的中文字符创建一个过滤后的词表，用来专门构造英文字符
    filtered_vocab = {key: vocab[key] for key in vocab.keys() if '中' not in key}
    #随机从字表选取sentence_length-1个字，可能重复
    x = [random.choice(list(filtered_vocab)) for _ in range(sentence_length-1)]
    # 生成一个随机索引
    random_index = random.randint(0, len(x))
    # 随即将中字中文字符插入列表中的某个位置
    x.insert(random_index, '中')
    # print("x is:{}".format(x))
    chinese_character_index = x.index('中')
    # 创建一个one-hot编码的向量，但只使用索引作为标签
    y_one_hot = np.zeros(sentence_length)
    y_one_hot[chinese_character_index] = 1
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    # print("y_one_hot is:{}".format(y_one_hot))
    return x, torch.tensor([chinese_character_index], dtype=torch.long) # 返回类别索引

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
def build_model(vocab, char_dim, sentence_length, num_classes):
    model = TextMulitClassifyModel(char_dim, sentence_length, vocab, num_classes)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        _, predicted = torch.max(y_pred, 1)  # 得到最大概率的索引Tensor，(batch_size, 1)
        correct = (predicted == y).sum().item() # 得到y_pred和y完全相等的元素的数量，即正确个数
        accuracy = correct / y.size(0)
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy

def main():
    #配置参数
    epoch_num = 20        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 25         #每个字的维度
    sentence_length = 6   #样本文本长度
    num_classes = sentence_length #类别个数
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample, vocab, sentence_length)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(int(train_sample // batch_size)):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
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
    char_dim = 25  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    num_classes = sentence_length #预测分类数
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length, num_classes)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    predicted_classes = torch.argmax(result, dim=1)
    probabilities = torch.max(result, dim=1).values

    for vec, pred_class, prob in zip(input_strings, predicted_classes, probabilities):
        # 将张量的元素转换为Python标量
        print("输入：%s, 预测中文字符元素出现的下标为：%d, 概率值：%f" % (vec, pred_class.item(), prob.item()))



if __name__ == "__main__":
    main()
    test_strings = ["fn中fee", "中wzsfg", "rg中deg", "nakww中"]
    predict("model.pth", "vocab.json", test_strings)