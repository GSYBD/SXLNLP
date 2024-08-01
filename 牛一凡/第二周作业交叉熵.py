import torch
import numpy as np

# 假设有3个样本每个样本在做4分类，此时这个样本是一个3*4维的张量。
# 根据描述分析4分类的样本相互互斥，每一份样本产生的概率随机无法确定， 但是4份样本的概率之和为1
# 通过已有的样本概率数据喂给机器，通过机器学习拟合出我需要的概率函数模型。

# 实现机器学习流程
# 1.输入人为设置好的样本数pred
# 2.机器学习端有个参数去接收输入的真实数据target
# 3.softmax函数先对接收的张量进行预处理，
# 3.1.初始化每份样本内的对应数据的比例值，此时每份样本内所有数据的数据之和等于1.
# 3.2.初始化后的每份样品对应项的比例是按照样本自己真实的数据大小*softmax数算的
# 3.3.此时输出完成后得到的数据就是对应每份样本内特定标签出现的比例了
# 4.to_one_hot初始化一个矩阵函数，在矩阵函数里学习对应的参数设置
# 4.1.矩阵参数里操作初始化参数全部展示为0，接入输入的值target，把对应参数值置为1，
# 4.2.矩阵记录了实际出现过的向量参数，类似学到了对应样本的内容。
# 5.完成对应参数的学习之后，调用交叉熵函数来记算损失误差.



# 真实存在的样本pred
pred = torch.FloatTensor([[5.7, 6.4, 7.1, 7.8],
                          [8.5, 9.2, 9.8, 10.6],
                          [11.3, 12.0, 12.7, 13.3]])
# 设置接收真实样本的个数（也可以叫维度数），现在是3个样本，所以接收的参数个数给了3个。参数里写的值是对应样本人为打的标签号
target = torch.LongTensor([1, 2, 3])
# 会用到softmax函数，先手动写个softmax函数，softmax函数做的就是把每个样本出现的概率给初始化出来
def softmax(matrix):
    return np.exp(matrix) / np.sum(np.exp(matrix), axis=1, keepdims=True)

print(torch.softmax(pred, dim=1))

# 会用到矩阵函数，先手动写矩阵函数初始化参数.
# target 传入的值，shape矩阵里原始就有的真实值
def to_one_hot(target,shape):
    # 对原始值全部初始化为0
    one_hot_target = np.zeros(shape)
    # 矩阵里传入的target参数有i个，进行i次循环，
    # 循环只进行如下操作：每次遇到有标识t的参数认为对应参数属于target，在矩阵来里把参数置为1.
    # 注此处的t可以是个数也可以是个数组或者向量
    for i,t in enumerate (target):
        one_hot_target[i][t] = 1
        return one_hot_target


# # 主体函数开始调用具体数值
# def cross_entropy (target, pred):
#     batch_size, class_num = pred.shape
#     pred = softmax(pred)
#     target = to_one_hot(target, pred.shape)
#     entropy = - np.sum(target * np.log(pred), axis=1)
#     return sum(entropy) / batch_size


# 主体函数开始调用具体数值，此处存疑问，我操作cross_entropy函数传入值的时候逻辑上先写target和先写pred的顺序没区别，
# 但是对应调用的时候pred.shape这行会报错，说少传了一个参数。但是实际查看softmax的打印值好像没少参数。就把对应参数顺序改回来了
def cross_entropy(pred, target):
    batch_size, class_num = pred.shape
    pred = softmax(pred)
    target = to_one_hot(target, pred.shape)
    entropy = - np.sum(target * np.log(pred), axis=1)
    return sum(entropy) / batch_size

print(cross_entropy(pred.numpy(), target.numpy()), "交叉熵输出误差")
