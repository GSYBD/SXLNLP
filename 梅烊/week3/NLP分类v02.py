# NLP分类
import numpy as np
import torch as torch
import json
import random

import matplotlib.pyplot as pl

# 生成一个样本
# sen_len：词向量的长度
# vocab：词汇表
def build_sample(vocab,sen_len):
    x = [random.choice(list("qrisgdafeklutyn你xiw")) for _ in range(sen_len)]
    x = np.array(x)

    # 注意，这里一定要用 np.array数组，否则不能达到预期
    # arr =np.array(['c','你', 'm', '你', 'd', 'y']) 
    # print( np.argwhere('你' == arr)[0] if np.argwhere('你' == arr).size != 0 else [-1])

    # 含字符“你”的句子，以“你”在字符串中的第一个索引值进行分类,不含字符“你”的句子被分类到 sen_len 类别
    # 实际的分类类别应该有（sen_len + 1）种，从0开始算
    y = np.argwhere('你' == x)[0][0] if np.argwhere('你' == x).size != 0 else sen_len  

    x = [vocab.get(word) for word in x] #将字转换成序号，为了做embedding
    # print(x, y)

    return x,y

# 生成一个批次样本
# sample_size 一个批次中样本的个数
# sen_len 样本的字符长度
def build_dataset(sample_size,sen_len):
    X=[]
    Y=[]    
    file_path="week3 深度学习处理文本/vocab.json"
    # file_path="downloads/vocab.json"  
    with open(file_path, "r",encoding="utf-8") as f:
        jsonStr= json.load(f)

    for i in range(sample_size):
        x,y = build_sample(jsonStr,sen_len)
        X.append(x)
        Y.append(y)
    return torch.tensor(X),torch.tensor(Y)


# 定义NLP分类模型
class NlpClassModel(torch.nn.Module):
    # vector_dim：词向量的长度
    # sentence_length：词语、句子的长度
    # vocab：词表
    def __init__(self, vector_dim,hidden_size,sen_len):
        super(NlpClassModel,self).__init__()        
        self.embedding = torch.nn.Embedding(29, vector_dim)

        # 使用rnn代替池化处理，rnn模型函数已经隐含了类似的池化处理
        # self.pool = torch.nn.AvgPool1d(sentence_length)
        self.rnnlayer = torch.nn.RNN(vector_dim, hidden_size, bias=False, batch_first=True)

        # linear层用于对rnn层的结果进行分类
        # sen_len : 输入的是rnn.output形状 N(batch_size)*L(sen_len)*hidden_size中的sen_len
        # sen_len + 1: 类别的个数
        self.classify = torch.nn.Linear(hidden_size, sen_len + 1)

        # 使用交叉熵计算线性层的损失值
        self.loss = torch.nn.functional.cross_entropy

    def forward(self, x,y = None):
        x = self.embedding(x)   # batch_size * sen_len => batch_size * sen_len * vector_dim 
        # print("embedding layer：",x)

        # x = x.transpose(1,2)    # batch_size * sen_len * vector_dim => batch_size * vector_dim * sen_len 
        # x= self.pool(x)            # batch_size * vector_dim * sen_len => batch_size * vector_dim * 1
        # print("after pool ===========",x)
        # x = x.squeeze()         # batch_size * vector_dim * 1 => batch_size * vector_dim
        
        # batch_size * sen_len * vector_dim => output:  batch_size * sen_len * hidden_size 
        # h:  batch_size * hidden
        output, h = self.rnnlayer(x)  
        # print("after rnn 预测值：",h)
        
        # 由于h.shape = (batch_size * hidden)，经rnn计算后少了sen_len的维度，需要对h进行维度重塑，以适配loss的运算
        h = h.squeeze()

        # output=output.transpose(hidden_size-2,hidden_size-1)
        # print("after transpose 预测值",output)

        # output = output.squeeze()
        # print("after squeeze 预测值",output)
        
        # output = output.squeeze()
        # print("after squeeze 预测值",output)

        # print("after classify 预测值：", output)
        # 以rnn层的最后一个hidden节点h的向量来表示一个样本
        # 以线性模型求h的预测值, batch_size * hidden_size => batch_size * out_features
        h = self.classify(h)
        # print("after classify 预测值：",h)
        
        if y is None:
            return h
        else:
            # print("实际值：",y)
            return self.loss(h,y)


# 对模型输出的一个预测值tensor进行转换，使其可以和sample的真实值进行对比
# y_pred_model.shape= (1, classify.out_features)
# 一维时使用dim=0，使用dim=1报错
# def transYpredForCompare(y_pred_model):
#     # print("pred_y before softmax:", y_pred_model)
#     # y =torch.nn.functional.softmax(y_pred_model, dim=0)
#     # print("pred_y after softmax:",y)
#     index_pred_y = torch.argmax(y).item()
#     print("y_pred.max.index :", index_pred_y)
#     return index_pred_y    
    

# 评估函数
# 用来评估每一轮训练的效果
def evaluate(model,sample_size, sen_len):
    model.eval()
    e_x,e_y = build_dataset(sample_size, sen_len)
    correct ,wrong =0,0    
    with torch.no_grad():
        pred_y = model(e_x)
        for p_y, t_y, t_x in zip(pred_y, e_y, e_x):
            # print("begin=======================================")
            # print("real x : ",t_x.tolist())
            # print("real y : ",t_y)
            # print("pred y ",p_y.tolist())
            # print("pred y max index :",torch.argmax(p_y))
            if torch.argmax(p_y).item() == t_y.item():
                correct +=1
            else:
                wrong +=1
            # print("end==========================================")
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 模型参数
    epoch = 20
    batch_size = 400

    # 每epoch的训练样本数
    train_size = 6000

    # 词表向量维度
    vector_dim = 5

    # 文本长度
    sen_len = 10

    # 隐藏层的个数
    # 进行NLP分类任务时，一个输入向量对应一个预测值的输出向量
    hidden_size = 10

    # 学习率
    learning_rate = 0.001

    # 创建模型
    model = NlpClassModel(vector_dim, hidden_size, sen_len)

    # 选定优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)


    train_x,train_y=build_dataset(train_size,sen_len)

    log =[]
    for i in range(epoch): 
        # 设置模型为训练模式
        model.train()

        watch_loss = []
        for batch_index in range(train_size//batch_size):
            # 构建train_size个样本，每个样本有sen_len个字符
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            # print(train_x,train_y)

            # 梯度归零
            optim.zero_grad()

            # 计算损失值
            loss = model.forward(x,y)

            # 计算梯度
            loss.backward()

            # 更新权重
            optim.step()

            watch_loss.append(loss.item())
            # end second for

        print("=========\n第%d轮平均loss:%f" % (i + 1, np.mean(watch_loss)))
        
        # 评估模型一轮训练的效果
        acc = evaluate(model,100,sen_len )
        
        log.append([acc,float(np.mean(watch_loss))])
        # end first for

    # torch.save(model.state_dict(),"downloads/nlp_rnn_model.pt")
    torch.save(model.state_dict(),"week3 深度学习处理文本/nlp_rnn_model.pt")
    pl.plot(range(len(log)),[l[0]  for l in log], label = 'acc')
    pl.plot(range(len(log)),[l[1] for l in log], label='loss')
    pl.legend()
    pl.show()

def predict(model_path , vector_dim, hidden_size, sen_len , test_samples):
    model = NlpClassModel(vector_dim=vector_dim, hidden_size=hidden_size,sen_len=sen_len)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result = model(test_samples)
    for input_vec, res in zip(test_samples, result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (np.array(input_vec),  torch.argmax(res),  res))

if __name__ == "__main__":
    main()

    # test_samples=torch.tensor([[ 9,  7,  4, 27,  1,  7, 18,  7,  6, 27],
    #     [12, 17, 18,  4,  6,  4, 20, 19,  6, 21],
    #     [17,  7, 23, 18, 18, 18, 23,  1, 11, 14],
    #     [24,  5, 14,  4,  4, 14, 12, 25,  1,  9],
    #     [17,  4, 19, 14, 20, 11, 18, 25, 18,  9],
    #     [24,  9, 12, 18, 21, 12, 17, 12,  4,  6],
    #     [14, 18, 21, 17, 19, 24, 21, 27, 25, 20],
    #     [25, 19, 21,  7, 19, 11, 17,  9, 25, 12],
    #     [ 1,  6, 19, 21,  5, 12, 24,  9, 27,  1]])
    # predict("downloads/nlp_rnn_model.pt",vector_dim,hidden_size,sen_len,test_samples)
    # predict("week3 深度学习处理文本/nlp_rnn_model.pt",vector_dim = 5,hidden_size =10 ,sen_len =10 ,test_samples = test_samples)

    # samples , y = build_dataset(9, 10)
    # print(samples)

    # arr =torch.tensor([23, 20, 17,  9,  4]) 
    # arr =np.array([23, 20, 17,  9,  4]) 
    # print(np.argwhere(3 == arr).size)
    # print( np.argwhere(10 == np.array(arr))[0] if np.argwhere(10 == np.array(arr)).size != 0 else -1)
    
        
          

