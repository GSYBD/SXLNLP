# -*- coding: utf-8 -*-
# name findpostitionweek3
# date 2024/6/30 16:29

import random
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


'''

输入字符串，根据字符A所在位置进行分类

'''

class TorchMdoel(nn.Module):
    def __init__(self,vector_dim,sentence_length,vocab):
        super(TorchMdoel,self).__init__()
        self.embedding = nn.Embedding(len(vocab),vector_dim) ## embedding 层
        # 能尝试切换RNN或者 pooling
        # self.pool = nn.AvgPoolld(sentence_length) # 池化层
        self.rnn = nn.RNN(vector_dim,vector_dim,batch_first = True)
        # + 1 的原因为可能出现A 不存在的情况，那是的真实lable在构造数据时设为了sentence_length
        self.classify = nn.Linear(vector_dim,sentence_length+1)
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值，无真实实标签，返回预测值
    def forward(self,x,y=None):
        x = self.embedding(x)
        # 使用polling的情况
        # x = xtransponse(1,2)
        # x = self.pool(x)
        # x = x.squeeze()
        rnn_out,hidden = self.rnn(x)
        x = rnn_out[:,-1, :] #或者hidden.squeeze() 也是可以的，因为RNN的hidden就是最后一个位置的输出

        # 接线性层做分类
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred,y) # 预测值和真实值计算损失
        else:
            return y_pred # 预测输出结果


    # 字符集随便挑一些字符
    #为 了每个字生成一个标号
    # {“a”:1,"b":2,"c":3}
    # abc ->[1,2,3]

def build_vocab():
    chars = "abcdefghijk"
    vocab = {"pad":0}
    for index ,char in enumerate(chars):
        vocab[char] = index + 1 # 每个字对应一个序号

    vocab['unk'] = len(vocab) #26
    return vocab

def build_sample(vocab,sentence_length):
    # 注意这里用sample是不放回的采样，每个字母不会重复出现，但 要求字符串长度要小于词表长度
    x = random.sample(list(vocab.keys()),sentence_length)
    # 指定哪些字出现时为正样本
    if 'a' in x:
        y = x.index('a')
    else:
        y = sentence_length
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x,y

# 建立数据集
# 输入需要的样本数量。需要多少可生成多少
def build_dataset(sample_length,vocab,sentence_length):
    dataset_x = []
    dataset_y = []
    for i in  range(sample_length):
        x , y  = build_sample(vocab,sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)

    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 建立模型
def build_model(vocab,char_dim,sentence_length):
    model = TorchMdoel(char_dim,sentence_length,vocab)
    return model


# 测试代码
# 测试模型的准确率
def evaluate(model,vocab,sample_length):
    model.eval()
    x,y = build_dataset(200,vocab,sample_length) # 建立220个用于测试的样本
    print("本次预测集中有%d个样本"%(len(y)))

    correct,wrong = 0,0
    with torch.no_grad():
        y_pred = model(x) #预测模型
        for y_p,y_t in zip(y_pred,y): # 与真实标签进行对比
            if int(torch.argmax(y_p)) == int(y_t):
                correct +=1
            else:
                wrong +=1

    print('正确预测个数：%d,正确率:%f'%(correct,correct/(correct+wrong)))


def main():
    # 配置参数
    epoch_num = 20 #训练轮数
    batch_size = 40 # 每次训练样本总数
    train_sample = 1000 # 每轮续联总共训练的样本总数
    char_dim = 30 # 每个字的维度
    sentence_length = 10 # 样本文本长度
    learning_rate = 0.001 # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model =build_model(vocab,char_dim,sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters() ,lr = learning_rate)

    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for bartch in range(int(train_sample / batch_size)):
            x,y = build_dataset(batch_size,vocab,sentence_length) # 构造一组训练样本
            optim.zero_grad() # 梯度归零
            loss = model(x,y) # 计算loss
            loss.backward() # 计算梯度
            optim.step() #g更新权重
            watch_loss.append(loss.item())

        print('-----第%d轮平均loss:%f'%(epoch +1,np.mean(watch_loss)))

        acc = evaluate(model,vocab,sentence_length) # 测试本轮模型结果
        log.append([acc,np.mean(watch_loss)])

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log] , label= 'acc') #画acc曲线
    plt.plot(range(len(log)), [l[0] for l in log], label ='loss') # 画loss曲线
    plt.legend()
    plt.show()


    # 保存模型
    torch.save(model.state_dict(),'model.pth')

    # 保存词表
    writer = open('vocab.json','w',encoding='utf8')
    writer.write(json.dumps(vocab,ensure_ascii=False,indent =2))
    writer.close()
    return


# 使用训练好的模型做预测

def predict(model_path,vocab_path,input_strings):

    char_dim = 30 # 每个字的维度
    sentence_length = 10 # 每个字文本长度
    vocab = json.load(open(vocab_path,'r',encoding = 'utf8'))
    model = build_model(vocab,char_dim,sentence_length)
    model.load_state_dict(torch.load(model_path))

    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string]) # 将输入序列化

    model.eval() ## 测试模式
    with torch.no_grad(): # 不计算梯度
        result= model.forward(torch.LongTensor(x)) # 模型预测

    for  i ,input_string in enumerate(input_strings):
        print("输入：%s,预测类别:%s ,概率值：%s"%(input_string,torch.argmax(result[i]), result[i]))  # 打印结果


if __name__ == "__main__":
    main()
    test_strings = ["kijabcdefh", "gijkbcdeaf", "gkijadfbec", "kijhdefacb"]
    predict('model.pth',"vocab.json",test_strings)