import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ps:突然理解为什么就要一个线性层。规律是最大值，那w非0且相等就完事了。妈的想了大半天,用了激活函数loss还大于1.

class TorchModel(nn.Module):
    def __init__(self,input_size):
        super(TorchModel,self).__init__()
        self.linaer = nn.Linear(input_size,5)
        # self.activation = nn.Softmax(dim=1)
        # self.activation = torch.sigmoid
        # self.out_linaer = nn.Linear(5,5)
        self.loss = nn.functional.cross_entropy
    
    def forward(self,x,y = None):
        y_pred = self.linaer(x)
        # x = self.activation(x)
        # y_pred = self.out_linaer(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred
    
# 生成样本
# 是个分类任务，样本x是五维向量，输出是个五位向量，输出样本中第一个最大值的idx，在y中为1，y其他值为0。
# sample: x = [1,2,5,3,1] , y_true = [0,0,1,0,0]
def build_sample():
    x = np.random.random(5)
    max_idx = np.argmax(x)
    y = np.zeros(5)
    y[max_idx] = 1
    return x,y
# 生成一批样本
def build_dataset(sample_num):
    X = []
    Y = []
    for i in range(sample_num):
        x,y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.FloatTensor(Y)


# 测试
# 暂时只要记住，训练调用model.train()；推理用model.eval()
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x,y = build_dataset(test_sample_num)
    # 计算每个类的总数
    class_counts = [torch.sum(y[:, i]).item() for i in range(y.size(1))]
    print("""
        当前预测集第1类总数为%d\n
        当前预测集第2类总数为%d\n
        当前预测集第3类总数为%d\n
        当前预测集第4类总数为%d\n
        当前预测集第5类总数为%d\n
        """% tuple(class_counts))
    correct, wrong = np.zeros(5),np.zeros(5)
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred,y):
            if np.argmax(y_p) == np.argmax(y_t):
                correct[np.argmax(y_t)] += 1
            else:
                wrong[np.argmax(y_t)] += 1
    print("正确预测个数：%d，正确率%f" %(np.sum(correct), np.sum(correct) / test_sample_num))
    return np.sum(correct) / test_sample_num


def main():
    print('main')
    # 训练参数
    epoch_num = 20
    batch_size = 20
    train_sample = 20000
    input_size = 5
    learning_rate = 0.001
    # model实例
    model = TorchModel(input_size)
    # 优化器？（传入w权重，学习率。）
    optim = torch.optim.Adam(model.parameters(),lr = learning_rate)
    log = []
    #创建训练集
    train_x, train_y = build_dataset(train_sample)
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
    torch.save(model.state_dict(),'model.pt')

    print(log)
    plt.plot(range(len(log)),[l[0] for l in log],label='acc')
    plt.plot(range(len(log)),[l[1] for l in log],label='loss')
    plt.legend()
    plt.show()
    return

def predict(model_path,input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    
    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec,result):
        print("输入:%s,预测类型：%d,概率值:%s"% (vec,np.argmax(res),res))


if __name__ == '__main__':
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.pt", test_vec)