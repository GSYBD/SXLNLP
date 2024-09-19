import torch
import torch.nn as nn
import numpy as np

'''
手动实现交叉熵的计算
'''

#使用torch计算交叉熵
ce_loss = nn.CrossEntropyLoss()
#假设有3个样本，每个都在做3分类
pred = torch.FloatTensor([[0.3, 0.1, 0.3],
                          [0.9, 0.2, 0.9],
                          [0.5, 0.4, 0.2]]) #n*class_num
#正确的类别分别为1,2,0
target = torch.LongTensor([1,2,0])     #n


loss = ce_loss(pred, target)
print(loss, "torch输出交叉熵")


#实现softmax函数
def softmax(matrix):
    return np.exp(matrix) / np.sum(np.exp(matrix), axis=1, keepdims=True)

#验证softmax函数
# print(torch.softmax(pred, dim=1))
# print(softmax(pred.numpy()))


#将输入转化为onehot矩阵
def to_one_hot(target, shape):
    one_hot_target = np.zeros(shape)
    for i, t in enumerate(target):
        one_hot_target[i][t] = 1
    return one_hot_target

#手动实现交叉熵
def cross_entropy(pred, target):
    batch_size, class_num = pred.shape
    pred = softmax(pred)
    target = to_one_hot(target, pred.shape)
    entropy = - np.sum(target * np.log(pred), axis=1)
    return sum(entropy) / batch_size

print(cross_entropy(pred.numpy(), target.numpy()), "手动实现交叉熵")

print(np.log(2.7))