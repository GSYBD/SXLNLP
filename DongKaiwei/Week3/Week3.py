# coding:utf-8

import json
import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


'''
Week3 Kivi
使用RNN完成一个简单的nlp任务

任务：
对于由6个字母构成序列s
如果首字母大写且其余字母中，大写字母更多，输出1
如果首字母大写且其余字母中，小写字母更多，输出2
如果首字母小写且其余字母中，大写字母更多，输出3
如果首字母小写且其余字母中，小写字母更多，输出4
'''

# RNN模型构造
class RNN(nn.Module):
    def __init__(self, vector_dim, hidden_dim, vocab, output_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=vocab['[pad]'])
        self.rnn = nn.RNN(vector_dim, hidden_dim, batch_first=True) # RNN层
        self.fc = nn.Linear(hidden_dim, output_dim) # 新增全连接层
        self.loss = nn.functional.cross_entropy # 交叉熵
        
    def forward(self, x, y=None):
        x = self.embedding(x) # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        output, _ = self.rnn(x)
        y_pred = self.fc(output[:, -1, :]) # 在RNN的输出上应用全连接层
        if y is not None:
            y = torch.squeeze(y)  # 将 y 转换为一维张量
            return self.loss(y_pred, y) # 预测值和真实值计算损失
        else:
            return y_pred # 输出预测结果
        
# 创建模型
def build_model(vocab, vector_dim=10, hidden_dim=20, output_dim=4):
    model = RNN(vector_dim, hidden_dim, vocab, output_dim)
    return model

# 生成字表
def build_vocab():
    # 获取脚本文件的绝对路径
    script_path = os.path.abspath(__file__)
    # 获取脚本文件所在的目录
    script_dir = os.path.dirname(script_path)
    # 改变工作目录到脚本文件所在的目录
    os.chdir(script_dir)
    with open('vocab.txt', 'r', encoding='utf-8') as f:
        chars = f.read()
    chars = sorted(set(chars))
    vocab = {'[pad]': 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['[unk]'] = len(vocab)
    # 保存字表
    writer = open('vocab.json', 'w', encoding='utf-8')
    writer.write(json.dumps(vocab, ensure_ascii=False, indent = 2))
    writer.close()
    return vocab

# 生成一个样本
def build_sample(vocab):
    chars = ''
    vocab_eff = vocab.copy()
    vocab_eff.pop('[pad]')
    vocab_eff.pop('[unk]')
    for i in range(5):
        chars += random.choice(list(vocab_eff.keys()))
    x = [vocab.get(word, vocab['[unk]']) for word in chars]
    y = label(chars)
    return x, y

# 自动判断chars的输出
def label(chars):
    if chars[0].isupper():
        if sum([1 for char in chars[1:] if char.isupper()]) > sum([1 for char in chars[1:] if char.islower()]):
            return 0
        else:
            return 1
    else:
        if sum([1 for char in chars[1:] if char.isupper()]) > sum([1 for char in chars[1:] if char.islower()]):
            return 2
        else:
            return 3

# 生成样本库
def build_dataset(vocab, num_samples):
    dataset_x = []
    dataset_y = []
    for i in range(num_samples):
        x, y = build_sample(vocab)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 模型每轮的准确率
def evaluate(model, vocab):
    model.eval()
    x, y = build_dataset(vocab, 100)   #建立100个用于测试的样本
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

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 10  # 每个字的维度
    hidden_dim = 20
    sort_numb = 4  # 分类数
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, hidden_dim, sort_numb)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, chars in enumerate(input_strings):
        print(f"输入：{chars}, 实际类别：{label(chars)}, 预测类别：{result[i].argmax()}") #打印结果


# 主程序

def main():
    #配置参数
    epoch_num = 30        #训练轮数
    batch_size = 50       #每次训练样本个数
    train_sample = 500    # 每轮训练总共训练的样本总数
    
    char_dim = 10         # 每个字的维度
    hidden_dim = 20 # 隐藏层维度
    sort_numb = 4   # 分类数
    
    learning_rate = 0.005 # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, hidden_dim, sort_numb)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(train_sample // batch_size):
            x, y = build_dataset(vocab, batch_size)   #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab)   #测试本轮模型结果
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

if __name__ == '__main__':
    main()
    test_strings = ["AAAAaa", "BbbBbb", "ccCCcC", "dddDdD"]
    predict("model.pth", "vocab.json", test_strings)