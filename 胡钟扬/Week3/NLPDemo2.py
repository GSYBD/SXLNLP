import torch

import torch.nn as nn

import numpy as np


import random


import json

import matplotlib.pyplot as plt

'''
作业目标：使用RNN进行多分类


'''

class TorchModel(nn.Module):
    
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size): 
        super(TorchModel,self).__init__()
        self.embedding = nn.Embedding(len(vocab),vector_dim, padding_idx=0) 
        
        # 可以理解成， 把词向量的维度降维成了hidden_size = 6
        self.rnn  = nn.RNN(input_size=vector_dim, hidden_size=hidden_size, 
                           bias=False, num_layers=2, batch_first=True)  
        
        self.bn=nn.BatchNorm1d(sentence_length)   # 往哪个维度的方向进行归一化就填哪个维度的数量
        self.pool = nn.AvgPool1d(sentence_length) 
        self.classify=nn.Linear(hidden_size,1)  
        self.activation = torch.sigmoid
        self.loss = nn.functional.mse_loss
    def forward(self, x:torch.Tensor, y=None):
        x = self.embedding(x)
        
        # print("embedding = \n", x)
        output, hidden = self.rnn(x)
        
        output # 20x6x7
        # print("rnn = \n", x)
        # print("output = \n",output)
        print("output.shape = \n",output.shape)

        print("--------------------------------------------------")
        # print("hidden = \n", hidden)
        print("hidden.shape = \n", hidden.shape)
        
        
        '''
            hidden: tensor of shape: (layers, N, H_{out}) containing the final hidden state
            for each element in the batch.
        '''
        
        # 将最后一层的 最后一个隐单元 作为RNN的输出
        hidden # 2x20x7  实际上是 2x20x6x7 ----> 每句话取最后一个隐单元
        
        # x = hidden[-1,:,:] # 20x7
        
        # print("x=\n",x)
        
        x= output
        
        x = self.bn(x)  # 20x6x7
        
        # print("bn = \n", x)
        print("bn.shape = \n", x.shape)
        
        x=x.transpose(1,2)  # 20x7x6
        x= self.pool(x)   # 20x7x1
        x = x.squeeze()   # 20x7
        x = self.classify(x)   # 
        x = self.activation(x)
        y_pred = x
        
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred




def build_vocab():
    vocab = {}
    vocab["pad"]=0
    for i, s in enumerate("abcdefghijklmnopqrstuvwxyz"):
        vocab[s] =i+1
    vocab["unk"] = len(vocab)
    return vocab
    




def build_sample(vocab:dict, sentence_length):
    x = [random.choice(list(vocab.keys())) for i in range(sentence_length)]
    
    # 指定正负样本判定规则
    '''
    & 是位运算符，在处理集合（如 set 或 frozenset）时，它代表交集操作。当应用于两个集合时，
    & 返回一个新集合，其中只包含同时存在于两个集合中的元素
    '''
    if set("abc") & set(x):
        y=1
    else:
        y=0
    
    # 每个字符转为序号
    x = [vocab.get(s, vocab["unk"]) for s in x]
    
    return x, y
    
    
    
    
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    
    for i in range(sample_length):
        x,y = build_sample(vocab, sentence_length)
        
        dataset_x.append(x)
        dataset_y.append([y])
    
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)





def main():
    #配置参数
    epoch_num = 20        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 5         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率
    
    hidden_size=7
    
    vocab = build_vocab()
    model = TorchModel(char_dim, sentence_length, vocab, hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample/batch_size)):         
            optimizer.zero_grad()
            batch_x, batch_y = build_dataset(batch_size, vocab, sentence_length)
            loss = model(batch_x, batch_y)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        
        acc = evaluate(model, vocab , sentence_length)
        avg_loss = np.mean(watch_loss)
        log.append((acc, avg_loss))
        

                
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    
    
    print("embedding weight",model.state_dict()["embedding.weight"].shape)
    print("RNN weight U-Layer1", model.state_dict()["rnn.weight_ih_l0"].shape)
    print("RNN weight W-layer1", model.state_dict()["rnn.weight_hh_l0"].shape)
    
    print("RNN weight U-Layer2", model.state_dict()["rnn.weight_ih_l1"].shape)
    
    print("RNN weight W-Layer2", model.state_dict()["rnn.weight_hh_l1"].shape)
    
    print("bn weight",model.state_dict()["bn.weight"].shape)
    print("classify weight",model.state_dict()["classify.weight"].shape)
    
    
    torch.save(model.state_dict(), "model.pt")
    
    
    writer = open('vocab.json', 'w', encoding='utf-8')
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()

    return

def build_model(vocab:dict, char_dim, sentence_length, hidden_size):
    model = TorchModel(char_dim, sentence_length, vocab, hidden_size)
    
    return model



def evaluate(model:TorchModel, vocab, sentence_length):
    model.eval()
    
    x, y  = build_dataset(200, vocab, sentence_length)
    x:torch.FloatTensor
    print(y.shape)
    print(f"总共有{x.size()}个样本，其中{torch.sum(y)}个正例，{len(x)-torch.sum(y.squeeze())}个反例")
    
    y_pred = model(x)
    
    correct, wrong =0, 0
    
    with torch.no_grad():
        for y_t, y_p in zip(y, y_pred):
            if y_p<0.5 and int(y_t)==0:
                correct+=1
            elif y_p>=0.5 and int(y_t)==1:
                correct+=1
            else:
                wrong+=1
    print(f"正确预测个数：{correct}， 正确率：{correct/(correct+wrong)*100}%")

    return correct/(correct+wrong)

def predict(model_weight_path, vocab_path, input_strings):
    # 加载vocab
    vocab:dict = json.load(open(vocab_path, 'r', encoding='utf-8'))

    model:TorchModel = build_model(char_dim=5,sentence_length=6,vocab=vocab, hidden_size=7)
    
    # 读取并设置模型权重
    model.load_state_dict(torch.load(model_weight_path))
    
    # print(model.state_dict()['embedding.weight'])
    

    
    sequences=[]
    
    print("vocab = \n", vocab)
    
    for input_string in input_strings:
        # 字符串转数字序列
        sequences.append([vocab[s] for s in input_string])
        
    model.eval()
    with torch.no_grad():
        y_pred=model(torch.LongTensor(sequences)) # LongTensor
        
        
    print(f"预测值是：",y_pred)
    
    for i, input_string in enumerate(input_strings):
        print(f"第{i}个字符序列是：{input_string}，所属类别是：{round(float(y_pred[i]))}，概率值是{y_pred[i]}")






if __name__ == '__main__':
    main()
    test_strings = ["fnvfee", "wzsdfg", "rqwdeg", "nakwww"]
    predict("model.pt", "vocab.json", test_strings)
    
    
    
    