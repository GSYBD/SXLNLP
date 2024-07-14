import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import string
import json
import os

class TorchModel(nn.Module):
    def __init__(self,vector_dim,hidden_size,vocab):
        super(TorchModel,self).__init__()
        self.embedding = nn.Embedding(len(vocab),vector_dim)
        self.layer = nn.RNN(vector_dim,hidden_size,batch_first=True)
        self.classify = nn.Linear(10,6)
        self.loss = nn.functional.cross_entropy
    
    def forward(self,x,y = None):
        x = self.embedding(x)
        _,out = self.layer(x)
        x = out.squeeze()
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred
    
# 生成样本
# 样本就是5个字符，输出y是6维向量。字符‘a’所在的idx，在y[idx] = 1。如果该样本没有'a',则y[5] = 1
# sample:
# 1. x = "bacde" y = [0,1,0,0,0,0]
# 2. x = "cdefg" y= [0,0,0,0,0,1]
def generate_random_string(length):
    characters = string.ascii_lowercase
    charA = 'a'
    random_string = ''
    for _ in range(length):
        if(random.random()<0.2):
            random_string += charA
        else:
            random_string += random.choice(characters)
    return random_string

def build_sample(str_len,vocab):
    x = np.array(str2seq(generate_random_string(str_len),vocab))
    y = np.zeros(6)
    idx = np.where(x == vocab['a'])
    if len(idx[0]) == 0:
        y[5] = 1
    else:
        y[idx[0][0]] = 1
    return x,y

# 生成一批样本
def build_dataset(sample_num,vocab,str_len):
    X = []
    Y = []
    for _ in range(sample_num):
        x,y = build_sample(str_len,vocab)
        X.append(x)
        Y.append(y)
        # 一定要转成floatTensor吗？
    return torch.IntTensor(X),torch.FloatTensor(Y)

# 测试
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x,y = build_dataset(test_sample_num,load_vocab(get_current_path('vocab.json')),5)

    correct, wrong = np.zeros(6),np.zeros(6)
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred,y):
            if np.argmax(y_p) == np.argmax(y_t):
                correct[np.argmax(y_t)] += 1
            else:
                wrong[np.argmax(y_t)] += 1
    print("正确预测个数：%d，正确率%f" %(np.sum(correct), np.sum(correct) / test_sample_num))
    return np.sum(correct) / test_sample_num

def load_vocab(path):
    with open(path,'r',encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab

def str2seq(str,vocab):
    return [vocab[s] for s in str]

def get_current_path(file = ''):
    current_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_path)
    return os.path.join(current_dir, file)

def main():
    print('main')
    # 训练参数
    epoch_num = 20
    batch_size = 20
    train_sample = 10000
    input_size = 10
    hidden_size = 10
    str_len = 5
    learning_rate = 0.001
    vocab = load_vocab(get_current_path('vocab.json'))
    
    # model实例
    model = TorchModel(input_size,hidden_size,vocab)
    # 优化器？（传入w权重，学习率。）
    optim = torch.optim.Adam(model.parameters(),lr = learning_rate)
    log = []
    #创建训练集
    train_x, train_y = build_dataset(train_sample,vocab,str_len)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) *batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) *batch_size]
            loss = model(x,y)
            # 计算梯度
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        loss_mean = float(np.mean(watch_loss))
        print('=========\n第%d轮的平均loss:%f'%(epoch+1,loss_mean))
        acc = evaluate(model)
        log.append([acc,loss_mean])
    
    # 保存模型
    torch.save(model.state_dict(),get_current_path('model.pt'))

    print(log)
    plt.plot(range(len(log)),[l[0] for l in log],label='acc')
    plt.plot(range(len(log)),[l[1] for l in log],label='loss')
    plt.legend()
    plt.show()
    return

def predict(model_path,input_vec):
    model = TorchModel(10,10,load_vocab(get_current_path('vocab.json')))
    model.load_state_dict(torch.load(model_path))
    
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.IntTensor(input_vec))
    print(input_vec)
    for vec, res in zip(input_vec,result):
        print("输入:%s,预测类型：%d,概率值:%s"% (vec,np.argmax(res),res))


if __name__ == '__main__':
    main()
    # test = []
    # model_path = get_current_path('model.pt')
    # vocab_path = get_current_path('vocab.json')
    # vocab = load_vocab(vocab_path)
    # test.append(str2seq('abcde',vocab))
    # test.append(str2seq('aacde',vocab))
    # test.append(str2seq('vacre',vocab))
    # test.append(str2seq('rbcdc',vocab))
    # test.append(str2seq('ibcdx',vocab))
    # predict(model_path,test)