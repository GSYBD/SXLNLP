from torchsummary import summary
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

class TorchRNNModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchRNNModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim)  #embedding层
        self.rnn_layer = nn.RNN(vector_dim, vector_dim, bias=True, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(vector_dim)
        self.rnn_layer2 = nn.RNN(vector_dim, 5, bias=True, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(5)
        self.pool = nn.AvgPool1d(sentence_length)   #池化层
        # self.classify = nn.Linear(vector_dim, 5)     #线性层
        self.activation = nn.Softmax(dim=1)
        self.criterion = nn.NLLLoss()

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)

        z, _ = self.rnn_layer(x)                      #(batch_size, sen_len, vector_dim) -> (batch_size, sen_len, vector_dim)
        z = self.layer_norm1(z)
        z, _ = self.rnn_layer2(z)
        z = self.layer_norm2(z)
        x = z.transpose(1, 2)                      #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        x = self.pool(x)                           #(batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        x = x.squeeze()                            #(batch_size, vector_dim, 1) -> (batch_size, vector_dim)

        # x = self.classify(x)                       #(batch_size, vector_dim) -> (batch_size, 1) 3*5 5*1 -> 3*1
        y_pred = self.activation(x)                #(batch_size, 1) -> (batch_size, 1)

        if y is not None:
            y = y.flatten()
            return self.criterion(torch.log(y_pred), y)
        else:
            return y_pred                 #输出预测结果

def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz我"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab) #26
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
#反之为负样本
def build_sample(vocab, sentence_length):
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length - 1)]
    if "我" in x:
        x.append("a")
    else:
        x.append("我")
    random.shuffle(x)
    number = x.index("我")
    if 0 <= number <= 2:
        y = 0
    elif 3 <= number <= 5:
        y = 1
    elif 6 <= number <= 8:
        y = 2
    elif 9 <= number <= 11:
        y = 3
    elif 12 <= number <= 14:
        y = 4
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本

    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        predicted_classes = torch.argmax(y_pred, dim=1)  # 获取最高概率的类别索引
        # print(predicted_classes)
        y_true = y.flatten()
        # print('x:', x)
        # print(y_true, y_pred)
        # print('y_pred:', predicted_classes)
        correct_predictions = (predicted_classes == y_true).sum().item()
    # print("预测类别：%s, 真实类别：%s" % (predicted_classes.numpy(), y_true))
    print("正确率：%f" % (correct_predictions / y_pred.shape[0]))
    return correct_predictions / y_pred.shape[0]


#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchRNNModel(char_dim, sentence_length, vocab)
    return model

def main():
    #配置参数
    epoch_num = 100        #训练轮数
    batch_size = 30       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 15   #样本文本长度
    learning_rate = 0.001 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
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
    torch.save(model.state_dict(), "model_rnn.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 15  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    predicted_classes = torch.argmax(result, dim=1)
    for i in range(len(predicted_classes)):
        print("输入文本序号：%s" % i)
        print("预测类别是：%s" % predicted_classes[i].item())



if __name__ == "__main__":
    # main()
    # label： 1   0   4  3
    test_strings = ["fnv我efnvfefnvfe", "我zsdfwzsdfwzsdf", "rqwderqwderqw我e", "nakwwnakw我naffa"]
    predict("model_rnn.pth", "vocab.json", test_strings)
