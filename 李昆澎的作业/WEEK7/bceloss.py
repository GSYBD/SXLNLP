import torch
import torch.nn as nn

#bce loss pytorch使用
sig = nn.Sigmoid()
input = torch.randn(5) #随机构造一个模型当前的预测结果 y_pred
input = sig(input)
target = torch.FloatTensor([1,0,1,0,1]) #真实label   y_true
bceloss = nn.BCELoss() #bce loss
loss = bceloss(input, target)
print(loss)

#自己模拟
l = 0
for x, y in zip(input, target):
    l += y*torch.log(x) + (1-y)*torch.log(1-x)
l = -l/5 #求和、平均、取负
print(l)


#对比交叉熵，区别在于交叉熵要求真实值是一个固定类别
# celoss = nn.CrossEntropyLoss()
# input = torch.FloatTensor([[0.1,0.2,0.3,0.1,0.3],[0.1,0.2,0.3,0.1,0.3]])
# target = torch.LongTensor([2,3])
# output = celoss(input, target)
# print(output)