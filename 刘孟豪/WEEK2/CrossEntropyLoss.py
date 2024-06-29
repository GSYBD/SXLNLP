import torch
import torch.nn as nn
import numpy as np

# torch计算交叉熵
loss = nn.CrossEntropyLoss()

pre_tensor = torch.FloatTensor([[0.5, 0.3, 0.4, 0.15], [0.1, 0.9, 0.3, 0.4], [0.3, 0.8, 0.2, 0.6]])

# 正确的类别为1，2，0--->[0,1,0],[0,0,1],[1,0,0]
target = torch.LongTensor([1, 2, 0])


# 手动计算交叉熵 预测值对应值作为真值，取loge（对数据标准化吗，防止熵值波动太大）,
# 和样本值对位相乘并相加，*-1（0-1为真数的对数，值恒定<0(底数>1),取反方便观察）

# 实现softmax函数，和相加为1
def softmax(a):
    return np.exp(a) / np.sum(np.exp(a), axis=1,
                              keepdims=True)  # axis=1除了第二个下标可以不同，其他下标必须都相同，keepdims=True保持结果的维度与原始array相同


# 将输入转换为onehot矩阵，1，2，0--->[0,1,0],[0,0,1],[1,0,0]
def to_one_hot(target, shape):
    zeros = np.zeros(shape)
    for i, t in enumerate(target):
        zeros[i][t] = 1
    return zeros

# 手动实现交叉熵
def cross_entropy(pre_tensor, target):
    # tensor.shape作为一个属性，用于获取张量的形状,class_name 是列，batch_size行
    batch_size, class_num = pre_tensor.shape
    # 归一化指数函数
    pre_tensor = softmax(pre_tensor)
    # 将输入转换为onehot矩阵
    target = to_one_hot(target, pre_tensor.shape)
    # 交叉熵运算
    np__sum = - np.sum(target * np.log(pre_tensor), axis=1)
    # 除以样本数
    return sum(np__sum) / batch_size

print(cross_entropy(pre_tensor.numpy(), target.numpy()), '手动实现交叉熵')
print(loss(pre_tensor, target), "torch输出交叉熵")
# 生成随机的样本
# def build_sample():
#     # 生成5维向量
#     x = np.random.random(5)
#     if x[0] > x[4]:
#         return x, 1
#     else:
#         return x, 0
#
#
# def build_sample_list(total_sample_num):
#     X = []
#     Y = []
#     for i in range(total_sample_num):
#         x, y = build_sample()
#         X.append(x)
#         Y.append([y])
#     return torch.FloatTensor(X), torch.FloatTensor(Y)
#
#
# print(build_sample_list(10))
