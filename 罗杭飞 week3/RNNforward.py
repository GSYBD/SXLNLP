#coding:utf8

import torch
import torch.nn as nn
import numpy as np


"""
手动实现简单的神经网络
使用pytorch实现RNN
手动实现RNN
对比
"""

class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TorchRNN, self).__init__()
        self.layer = nn.RNN(input_size, hidden_size, bias=False, batch_first=True)

    def forward(self, x):
        return self.layer(x)

#自定义RNN模型
class DiyModel:
    def __init__(self, w_ih, w_hh, hidden_size):
        self.w_ih = w_ih
        self.w_hh = w_hh
        self.hidden_size = hidden_size

    def forward(self, x):
        ht = np.zeros((self.hidden_size))
        output = []
        for xt in x:
            ux = np.dot(self.w_ih, xt)
            wh = np.dot(self.w_hh, ht)
            ht_next = np.tanh(ux + wh)
            output.append(ht_next)
            ht = ht_next
        return np.array(output), ht


x = np.array([[1, 2, 3],
              [3, 4, 5],
              [5, 6, 7]])  #网络输入

#torch实验
hidden_size = 4
torch_model = TorchRNN(3, hidden_size)

# print(torch_model.state_dict())
w_ih = torch_model.state_dict()["layer.weight_ih_l0"]
w_hh = torch_model.state_dict()["layer.weight_hh_l0"]
print(w_ih, w_ih.shape)
print(w_hh, w_hh.shape)
#
torch_x = torch.FloatTensor([x])
output, h = torch_model.forward(torch_x)
print(h)
print(output.detach().numpy(), "torch模型预测结果")
print(h.detach().numpy(), "torch模型预测隐含层结果")
print("---------------")
diy_model = DiyModel(w_ih, w_hh, hidden_size)
output, h = diy_model.forward(x)
print(output, "diy模型预测结果")
print(h, "diy模型预测隐含层结果")
