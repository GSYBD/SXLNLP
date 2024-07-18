import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
def build_vocab():
    build_vocab_dic ={}
    vocab_list ="你abcdefghijklmn"
    build_vocab_dic['pad'] = 0
    for  i,r  in  enumerate(vocab_list):
        build_vocab_dic[r]=i+1
    build_vocab_dic['unk']=len(vocab_list)+1
    json_data = json.dumps(build_vocab_dic, indent=4)
    with open('vocab1.json', 'w') as json_file:
        json_file.write(json_data)
    return build_vocab_dic

def creat_datas(zi_size,build_vocab_dic,batch_size):
    data_x = []
    data_y = []
    str_a_li = []
    for i in range(batch_size):
        x_li = [random.choice(list(build_vocab_dic.keys())) for _ in range(zi_size)]
        if set(x_li) & set('你'):
            pass
        else:
            x_li[random.randint(0,zi_size-1)]='你'
        str_a_li.append(''.join(i for i in x_li))
        a = np.zeros(zi_size)
        a[x_li.index('你')] = 1
        x = [build_vocab_dic[i1]  for i1 in  x_li]
        data_x.append(x)
        data_y.append(a)
    return torch.LongTensor(data_x),torch.FloatTensor(data_y),str_a_li

# nlp的 rnn 网络层级
class  TorchModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_size,output_size=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  #embedding层
        self.layer = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # 修改为输出类别数量
        # self.activation = torch.softmax     #sigmoid归一化函数
        self.bn = nn.BatchNorm1d(output_size)  # 根据 hidden_size 初始化

    def forward(self,x):
        x = self.embedding(x)
        x, _ = self.layer(x)  # 假设使用RNN层
        # print('1',x.shape)
        # x = self.dropout(x)  # 应用Dropout
        x = x[:, -1, :]
        # print('2',x.shape)
        x = self.fc(x)  # 取序列的最后一个时间步
        # print('3',x[0],x.shape)
        pre_y =self.bn(x)
        # print(x[0],x.shape)
        # pre_y = self.activation(x)
        # print(pre_y.shape)
        return pre_y


def  data_run(zi_size,batch_size,file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        # 加载JSON数据
        build_vocab_dic = json.load(file)
    data_x,data_y,str_a_li   =creat_datas(zi_size,build_vocab_dic,batch_size)
    return  data_x,data_y,str_a_li,build_vocab_dic
def evaluate(model):
    model.eval()
    with torch.no_grad():
        zi_size = 10  # 每个字有多少维代替
        batch_size = 100  # 多少 组 样本
        file_name = 'vocab1.json'
        test_x, test_y, str_a, build_vocab_dic = data_run(zi_size, batch_size, file_name)
        output = model(test_x)
        correct = 0
        wrong = 0
        for y_p, y_t in zip(output, test_y):
            predicted_index = torch.argmax(y_p).item()
            predicted_index1 = torch.argmax(y_t).item()
            if predicted_index == predicted_index1:
                correct += 1
            else:
                wrong += 1
        # for y_p, y_t in zip(output, test_y):
        #     if float(y_p) <0.5 and  int(y_t) ==0:
        #         correct += 1
        #     elif float(y_p) >=0.5 and  int(y_t) ==1:
        #         correct += 1
        #     else:
        #         wrong+=1
        print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))

def run():
    # 数据生成
    zi_size = 10  # 每个字有多少维代替
    batch_size = 1000  # 多少 组 样本
    file_name = 'vocab1.json'
    data_x,data_y,str_a,build_vocab_dic = data_run(zi_size,batch_size,file_name)
    vocab_size = len(build_vocab_dic)  # 词汇表大小
    batch_size = 1000  # 多少 组 样本
    embedding_dim = 256  # 嵌入维度
    hidden_size = 256  # RNN隐藏层大小
    learning_rate =0.03
    batch_size = 128
    epochs = 10
    model = TorchModel(vocab_size, embedding_dim, hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for   i in  range(epochs):
        model.train()
        loss_li=[]
        total_num = len(data_x)
        for j in range(total_num//batch_size):
            tran_x1 = data_x[j*batch_size:(j+1)*batch_size]
            tran_y1 = data_y[j*batch_size:(j+1)*batch_size]
            pre_y = model(tran_x1)
            loss = criterion(pre_y, tran_y1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()  # 清空梯度
            loss_li.append(loss.item())
        evaluate(model)
        print(f"第{i}轮的loss:{np.mean(loss_li)}")
    torch.save(model.state_dict(), "model你.pt")
def predict(model_path,batch_size=10):
    zi_size = 10  # 每个字有多少维代替  # batch_size是多少 组 样本
    file_name = 'vocab1.json'
    pred_x,pred_y, str_a_li, build_vocab_dic = data_run(zi_size, batch_size, file_name)
    vocab_size = len(build_vocab_dic)  # 词汇表大小
    embedding_dim = 256  # 嵌入维度
    hidden_size = 256  # RNN隐藏层大小
    model = TorchModel(vocab_size, embedding_dim, hidden_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        pre_y =model(pred_x)
        for i, j, z, z1 in zip(pre_y, pred_x, pred_y, str_a_li):
            print(i, '\n', z1, '\n', j, z)

if __name__ == '__main__':
    # run()
    # 数据生成
    model_path = 'model你.pt'
    predict(model_path,batch_size=5)

