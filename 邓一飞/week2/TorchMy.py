
import torch
import torch.nn as nn
import numpy as np

"""
使用torch 模型预测一个分类任务
在一个1X5矩阵重查找最小值在第几位就归为第几位
如:
[1,2,3,4,5] : [1,0,0,0,0] 第1类
[2,1,3,4,5] : [0,1,0,0,0] 第2类
[6,3,2,4,5] : [0,0,1,0,0] 第3类
[8,4,7,2,5] : [0,0,0,1,0] 第4类
[3,2,4,5,1] : [0,0,0,0,1] 第5类
"""
class  myTorchModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2):
        super(myTorchModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 5) # 5*3
        #self.layer2 = nn.Linear(hidden_size1, hidden_size2) # 3*5
        # 添加激活函数
        #self.sigmoid = nn.Sigmoid()
        # 添加softmax激活函数
        #self.softmax = nn.Softmax(0)
        # 添加损失函数
        self.loss = nn.CrossEntropyLoss()


    def forward(self, x,y=None):
        x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.sigmoid(x)
        #x = self.softmax(x)
        if y is None:
            #return self.softmax(x)
            return x
        x = self.loss(x, y)
        return x

#构造一条数据
def randomOneData(num):
    rr = np.random.random(num)
    min = np.min(rr)
    yy = np.zeros(num)
    for i,t in enumerate(rr):
        if t == min:
            yy[i] = 1
    # print(rr, yy)
    return rr,yy

#构造一些数据
def buildRandomData(num):
    datax = np.zeros((num,5))
    datay = np.zeros((num,5))
    for t in range(num):
        xx,yy = randomOneData(5)
        datax[t] = xx
        datay[t] = yy
    return datax,datay



#训练数据
def testEvaluation(model):
    model.eval()
    t_x, t_y = buildRandomData(100)
    count = 0
    with torch.no_grad():
        yp = model.forward(torch.FloatTensor(t_x))
        #和Y 比较统计猜对的正确率。
        for i,t in enumerate(t_y):
            if np.argmin(yp[i].detach().numpy()) == np.argmin(t_y[i]):
                count += 1
    return count/len(t_y)




def main():
    epoch_num = 1000
    batch_size = 20
    data_num = 5000

    learning_rate = 0.01

    t_x,t_y = buildRandomData(data_num)

    input_size = 5
    hidden_size1 = 3
    hidden_size2 = 5
    model = myTorchModel(input_size, hidden_size1, hidden_size2)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(data_num/batch_size)):
            x = torch.FloatTensor(t_x[batch*batch_size:(batch+1)*batch_size])
            y = torch.Tensor(t_y[batch*batch_size:(batch+1)*batch_size])
            loss = model.forward(x,y) # 计算损失
            loss.backward() # 计算梯度
            #print(loss)
            optim.step()  # 更新权重
            #optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("epoch:%d,loss:%f" %(epoch, np.mean(watch_loss)))
        acc = testEvaluation(model)
        print("acc:%f" %(acc))

    # 保存模型
    torch.save(model.state_dict(), "model.pt")

def predict(model_path, input_vec):
    input_size = 5
    hidden_size1 = 3
    hidden_size2 = 5
    model = myTorchModel(input_size, hidden_size1, hidden_size2)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    #print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        #print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果
        yp = [0, 0, 0, 0, 0]
        yt = [0, 0, 0, 0, 0]
        ip = np.argmin(res.detach().numpy())
        it = np.argmin(vec)
        yp[ip]=1
        yt[it]=1
        print("输入：%s, 预测类别：%s, 真实类别：%s,%s,计算值：%s" % (vec, yp,yt,ip==it, res))

if __name__ == '__main__':
    main()
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # test_vec, test_y = buildRandomData(100)
    # print(test_vec)
    # print(test_y)
    # print("=====")
    # predict("model.pt", test_vec)