import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import torch.nn.functional

# 任务要求
# 自己设计文本任务目标，使用rnn进行多分类
# 任务目标 寻找 a 字符，如果句子中存在 a 字符，则输出a的位置，否则输出 -1.(分类长度为字典长度 + 1)
#############################
# 全局变量
#############################


class NLPWorkModule( nn.Module ) :
    # 字维度，句子长度，字典
    def __init__( self, vector_dim, sentence_length, vocab):
        super().__init__()
        self.embedding = nn.Embedding( len(vocab), vector_dim, padding_idx=0) #embedding层 将字典转换成向量
        # self.pool  = nn.AvgPool1d(sentence_length) #池化层,减少抖动
        # batch_first 第一维是batch_size
        self.rnn = nn.RNN( input_size=vector_dim, hidden_size=vector_dim, num_layers=1, batch_first=True ) #RNN input_size输入 hidden_size输出
        self.layer = nn.Linear( vector_dim, sentence_length + 1 )
        self.loss = torch.nn.functional.cross_entropy # 交叉熵损失函数
        
    def forward( self , x,y=None) :
        x = self.embedding(x)     #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        # x = x.transpose(1,2)    #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)       
        # x = self.pool(x)        #(batch_size, vector_dim, sen_len)->(batch_size, vector_dim, 1)
        # x = x.squeeze()         #(batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        # x = self.rnn(x)         #(batch_size, vector_dim) -> (batch_size, vector_dim)
        # x = self.rnn(x)         #(batch_size, vector_dim) -> (batch_size, 1) 3*5 5*1 -> 3*1
        # y_pred = self.activation(x)                #(batch_size, 1) -> (batch_size, 1)
        output,hidden = self.rnn(x)
        x = output[:,-1,:]
        y_pred = self.layer(x)
        print("y:{}",y )
        print("y_pred:{}",format(y_pred))
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred

# 创建字符表
def build_vocab():
    # 准备我的字符集
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    vocab = {"pad":0} 
    for index,char in enumerate(chars):
        vocab[char] = index+1
    vocab['unk'] = len(vocab) 
    return vocab

# 创建样本
# in 字典，字节长度
def build_sample(vocab, sentence_length):
    x = random.sample(list(vocab.keys()), sentence_length)
    if "a" in x :  
        y = x.index("a")
    else :
        y = sentence_length
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x,y

# 创建数据集 样本数量，字典，句子长度
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 创建模型 字典 字符 句子长度
def build_model( vocab, char_dim, sentence_length ):
    model = NLPWorkModule( char_dim, sentence_length, vocab )
    return model

#用来测试每轮模型的准确率
# 模型，字典，样本长度
def evaluate(model,vocab,sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)   #建立200个用于测试的样本
    print("本次预测集中共有%d个样本"%(len(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model.forward(x)
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if int(torch.argmax(y_p))== int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))    
    return correct/(correct+wrong)
        
def main():
    #配置参数
    epoch_num = 20          # 训练轮数
    batch_size = 20         # 每批样本数
    char_dim = 10           # 字向量维度
    sentence_length = 10    # 句子长度
    train_sample = 1000     # 样本数量   
    learning_rate = 0.005   # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log=[]
    # 训练轮次
    for epoch in range(epoch_num):
        # 开启训练
        model.train()
        # 梯度损失函数
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            # 建立数据集
            dataset_x, dataset_y = build_dataset(batch_size,vocab,sentence_length)
            optim.zero_grad()    #梯度归零
            loss = model.forward(dataset_x,dataset_y)    # 计算损失函数
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
            print("======\n 第%d次学习平均损失函数loss:%f \n" %( epoch + 1, np.mean(watch_loss)) )
            acc=evaluate(model,vocab,sentence_length)
            log.append([acc,np.mean(watch_loss)])
    #画图
    plt.plot(range(len(log)),[l[0] for l in log],label="acc")
    plt.plot(range(len(log)),[l[1] for l in log],label="loss")
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
# def predict( model_path, vocab_path, input_strings ):
#     char_dim = 30           # 字数的维度
#     sentence_length = 10    # 句子的长度
#     vocab = json.load(open(vocab_path,"r", encoding="utf8")) 
#     model = build_model(vocab, char_dim, sentence_length)    #建立模型
#     model.load_state_dict(torch.load(model_path))    
#     x=[]
#     for input_string in input_strings:
#         x.append([vocab[char] for char in input_string])     #将输入序列化
#     model.eval()
#     with torch.no_grad():
#         result = model.forward(torch.LongTensor(x))
#     for i, input_string in enumerate(input_strings):
#         print("输入%s 预测类别%s 概率%s" %(input_string, torch.argmax(result[i]), result[i] ) )

def predict(model_path, vocab_path, input_strings):
    char_dim = 30  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, torch.argmax(result[i]), result[i])) #打印结果

if __name__ == '__main__' :
    main()
    test_strings = ["kijabcdefh", "gijkbcdeaf", "gkijadfbec", "kijhdefacb"]
    predict( "model.pth","vocab.json",test_strings)
    