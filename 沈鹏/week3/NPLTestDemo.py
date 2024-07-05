import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

# 基于pytorch的网络编写
# 实现一个网络完成简单的nlp任务
# 判断文本中是否有某种特定字符出现

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_lenght,vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim) # embedding层
        self.pool = nn.AvgPool1d(sentence_lenght) #池化层
        self.classify = nn.Linear(vector_dim, 1) #线性层
        self.activation = torch.sigmoid # simgoid 归一化函数
        self.loss = nn.functional.mse_loss # loss损失函数采用均方差损失

    def forward(self, x, y = None):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.squeeze()
        x = self.classify(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y) #预测值和真实值计算损失
        else:
            return y_pred  #输出预测结果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def create_char_dic():
    chars = "abcdefghijklmnopqrstuvwxyz" # 字符集
    dic = {"pad": 0}
    #enumerate 遍历字符
    for index, char in enumerate(chars):
        dic[char] = index + 1
    dic["unk"] = len(dic) # 最后一位上加上位置别标志
    return dic


def build_sample(char_dic, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(char_dic.keys())) for _ in range(sentence_length)]
    # 指定哪些字出现为正
    if set("abc") & set(x):
        y = 1
    else:
        y = 0
    x = [char_dic.get(word, char_dic) for word in x] # 将字转换成序号，为了做embedding
    return x, y

# 创建数据集
# 输入需要的样本数量，需要生成多少
def create_datas(sample_length, char_dic, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(char_dic, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)
def create_model(char_dic, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, char_dic)
    return model

#测试代码
# 用来测试每轮模型的准确率
def evaluate(model, chat_dic, sample_length):
    model.eval()
    x, y = create_datas(200, chat_dic, sample_length) #建立200个用于测试的样本
    print("本次预测集中共有%d个正样本，%d个负样本" %(sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x) # 模型预测
        for y_p, y_t in zip(y_pred, y): # 与真实标签进行对比
            if float(y_p) < 0.5 and int(y_t) == 0:
                correct += 1
            elif float(y_p) > 0.5 and int(y_t) == 1:
                correct += 1
            else:
                wrong += 1
    print("正确预测的个数：%d, 正确率：%f"%(correct,correct/(correct + wrong)))

def main():
    # 配置参数
    epoch_num = 20 # 训练轮数
    bach_size = 20 # 每次训练样本个数
    train_sample = 500 # 每轮训练总共训练的样本总数
    char_dim = 20 # 每个字的维度
    sentence_length = 6 # 样本文本的长度
    learning_rate = 0.005 # 学习率
    # 建立字表
    char_dic = create_char_dic()
    # 建立模型
    model = create_model(char_dic, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample/bach_size)):
            x, y = create_datas(bach_size, char_dic, sentence_length)  #构造一组训练样本
            optim.zero_grad # 梯度归零
            loss = model(x, y) #计算loss
            loss.backward()  #计算梯度
            optim.step() # 更新权重
            watch_loss.append(loss.item())
        print("\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, char_dic, sentence_length)
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
    writer.write(json.dumps(char_dic, ensure_ascii=False, indent=2))
    writer.close()
    return

if __name__ == '__main__':
    main()