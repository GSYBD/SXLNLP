import torch
import torch.nn as nn

'''
手动实现交叉熵的计算
'''

#使用torch计算交叉熵
ce_loss = nn.CrossEntropyLoss()
#假设有3个样本，每个都在做3分类
pred = torch.FloatTensor([[0.3, 0.1, 0.3],
                          [0.9, 0.2, 0.9],
                          [0.5, 0.4, 0.2],
                          [0.8, 0.1, 0.1],
                          [0.7, 0.4, 0.1]]) #n*class_num
#正确的类别分别为1,2,0
target = torch.LongTensor([1,2,0,1,2])     #n


loss = ce_loss(pred, target)
print(loss, "torch输出交叉熵")


