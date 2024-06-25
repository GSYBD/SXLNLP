#定义模型
import os
import torch
import torch.nn as nn
from dataset01 import mydataloader, eval_dataloader
from lib import max_len,embedding_dim , epochs,ws,hidden_size,num_layers,bidirectional
import lib

class jqModel(nn.Module):
    def __init__(self,input):
        super(jqModel, self).__init__()
        self.embedding = nn.Embedding(len(ws), embedding_dim)
        #加入LSTM模型
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,
                            num_layers=num_layers,bidirectional=bidirectional,
                            batch_first=True,dropout=lib.drop_out)
        self.linear = nn.Linear(hidden_size*2, 2)

    def forward(self, input):
        embedded = self.embedding(input) #输入为一个 batch_size x seq_len 的 tensor，转化后（batch_size,seq_len ,emdeding_dim）
        x,(h_n,c_n)=self.lstm(embedded) #x为(batch_size,seq_len,hidden_size*2) h_n为(num_layers*num_directions,batch_size,hidden_size)
        #c_n为(num_layers*num_directions,batch_size,hidden_size)
        #取最后一个时间点的输出x = x[:, -1, :]
        #双向lstm为第一个和最后一个进行concat操作
        output_fw = h_n[-2,:,:]#正向最后一次
        output_bw = h_n[-1,:,:]#反向最后一次
        output = torch.cat([output_fw,output_bw],dim=-1)#output为(batch_size,hidden_size*2)
        out = self.linear(output)
        out = nn.functional.sigmoid(out)
        return out
model = jqModel(input)
y_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
def train (epochs):
    model.train()
    if os.path.exists("./model.pkl"):
        model.load_state_dict(torch.load("./train_model.pkl"))
        optimizer.load_state_dict(torch.load("./optimizer.pkl"))
    for epoch in range(epochs):
        for i, (laber, con) in enumerate(mydataloader):
            optimizer.zero_grad()
            outputs = model(con)
            loss = y_loss(outputs,laber)
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print("第{}轮，第{}个batch，loss:{}".format(epoch+1,i+1,loss.item()))

def eval():
    model.eval()
    loss_list=[]
    acc_list=[]
    with torch.no_grad():
        for i, (laber, con) in enumerate(eval_dataloader):
            outputs = model(con)
            cur_loss = y_loss(outputs,laber)
            loss_list.append(cur_loss.item())
#           计算准确率predicted = torch.max(outputs,dim=1)
            y_ped = outputs.max(dim=-1)[-1]
            acc =y_ped.eq(laber).float().mean()
            acc_list.append(acc.item())
    print(outputs,acc_list)

if __name__ == '__main__':
#    train(epochs)
#    torch.save(model.state_dict(), "./train_model.pkl")
#    torch.save(optimizer.state_dict(), "./optimizer.pkl")
    eval()
