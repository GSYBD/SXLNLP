import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

'''
给定五个随机数数组，返回对应大小顺序数组，如[0.2,0.4,0.1,0.3,0.5]
排序后[(索引为:1)0.1 (索引为:2)0.2 (索引为:3)0.3 (索引为:4)0.4 (索引为:5)0.5] 
返回为[2,4,1,3,5]
'''

class TorchModel(nn.Module):
    def __init__(self,input_size):
        super(TorchModel,self).__init__()
        self.linear = nn.Linear(input_size, 5)
        self.activation = torch.sigmoid
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x, y=None):
        x = self.linear(x)
        y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

def build_sample():
    x = np.random.random(5)
    return x,getArrSortIndex(x)

def getArrSortIndex(arr):
    y = np.sort(arr)
    used = {}
    z = []
    for xv in arr:
        for index,value in enumerate(y):
            if xv == value:
                if index in used:
                    continue;
                else:
                    z.append(index)
                    break
    return z

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.FloatTensor(Y)

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p,y_t in zip(y_pred, y):
            y_ts = getArrSortIndex(y_t)
            y_ps = getArrSortIndex(y_p)
            j=0
            for i in range(5):
                if y_ps[i] == y_ts[i]:
                    j += 1
            if j==5:
                correct += 1
            else:
                wrong += 1
    print("预测正确个数：%d,正确率：%f" % (correct, correct/test_sample_num))
    return correct/test_sample_num

def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict)

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        print("输入:%s,预测：%s" % (vec, res))
    

def main():
    epoch_num = 20
    batch_size = 200
    train_sample = 50000
    input_size = 5
    learning_rate = 0.001
    model = TorchModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x,train_y = build_dataset(train_sample)
    for epoch in range(epoch_num):
        model.train
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    torch.save(model.state_dict(), "model.pt")
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

def predit_test():
    test_vec = []
    for i in range(5):
        test_vec.append(build_sample()[0][0])
    print(test_vec)
    predict("model.pt",test_vec)

if __name__ == "__main__":
    #main()
    predit_test()
