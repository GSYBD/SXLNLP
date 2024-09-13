
import torch
import torch.nn as nn
import numpy as np

'''
用矩阵运算的方式复现一些基础的模型结构
清楚模型的计算细节，有助于加深对于模型的理解，以及模型转换等工作
'''

#构造一个输入
length = 6
input_dim = 12
hidden_size = 7
x = np.random.random((length, input_dim))
# print(x)

#使用pytorch的lstm层
torch_lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
for key, weight in torch_lstm.state_dict().items():
    print(key, weight.shape)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

#将pytorch的lstm网络权重拿出来，用numpy通过矩阵运算实现lstm的计算
def numpy_lstm(x, state_dict):
    weight_ih = state_dict["weight_ih_l0"].numpy()
    weight_hh = state_dict["weight_hh_l0"].numpy()
    bias_ih = state_dict["bias_ih_l0"].numpy()
    bias_hh = state_dict["bias_hh_l0"].numpy()
    #pytorch将四个门的权重拼接存储，我们将它拆开
    w_i_x, w_f_x, w_c_x, w_o_x = weight_ih[0:hidden_size, :], \
                                 weight_ih[hidden_size:hidden_size*2, :],\
                                 weight_ih[hidden_size*2:hidden_size*3, :],\
                                 weight_ih[hidden_size*3:hidden_size*4, :]
    w_i_h, w_f_h, w_c_h, w_o_h = weight_hh[0:hidden_size, :], \
                                 weight_hh[hidden_size:hidden_size * 2, :], \
                                 weight_hh[hidden_size * 2:hidden_size * 3, :], \
                                 weight_hh[hidden_size * 3:hidden_size * 4, :]
    b_i_x, b_f_x, b_c_x, b_o_x = bias_ih[0:hidden_size], \
                                 bias_ih[hidden_size:hidden_size * 2], \
                                 bias_ih[hidden_size * 2:hidden_size * 3], \
                                 bias_ih[hidden_size * 3:hidden_size * 4]
    b_i_h, b_f_h, b_c_h, b_o_h = bias_hh[0:hidden_size], \
                                 bias_hh[hidden_size:hidden_size * 2], \
                                 bias_hh[hidden_size * 2:hidden_size * 3], \
                                 bias_hh[hidden_size * 3:hidden_size * 4]
    w_i = np.concatenate([w_i_h, w_i_x], axis=1)
    w_f = np.concatenate([w_f_h, w_f_x], axis=1)
    w_c = np.concatenate([w_c_h, w_c_x], axis=1)
    w_o = np.concatenate([w_o_h, w_o_x], axis=1)
    b_f = b_f_h + b_f_x
    b_i = b_i_h + b_i_x
    b_c = b_c_h + b_c_x
    b_o = b_o_h + b_o_x
    c_t = np.zeros((1, hidden_size))
    h_t = np.zeros((1, hidden_size))
    sequence_output = []
    for x_t in x:
        x_t = x_t[np.newaxis, :]
        hx = np.concatenate([h_t, x_t], axis=1)
        # f_t = sigmoid(np.dot(x_t, w_f_x.T) + b_f_x + np.dot(h_t, w_f_h.T) + b_f_h)
        f_t = sigmoid(np.dot(hx, w_f.T) + b_f)
        # i_t = sigmoid(np.dot(x_t, w_i_x.T) + b_i_x + np.dot(h_t, w_i_h.T) + b_i_h)
        i_t = sigmoid(np.dot(hx, w_i.T) + b_i)
        # g = np.tanh(np.dot(x_t, w_c_x.T) + b_c_x + np.dot(h_t, w_c_h.T) + b_c_h)
        g = np.tanh(np.dot(hx, w_c.T) + b_c)
        c_t = f_t * c_t + i_t * g
        # o_t = sigmoid(np.dot(x_t, w_o_x.T) + b_o_x + np.dot(h_t, w_o_h.T) + b_o_h)
        o_t = sigmoid(np.dot(hx, w_o.T) + b_o)
        h_t = o_t * np.tanh(c_t)
        sequence_output.append(h_t)
    return np.array(sequence_output), (h_t, c_t)


torch_sequence_output, (torch_h, torch_c) = torch_lstm(torch.Tensor([x]))
numpy_sequence_output, (numpy_h, numpy_c) = numpy_lstm(x, torch_lstm.state_dict())

print(torch_sequence_output)
print(numpy_sequence_output)
print("--------")
print(torch_h)
print(numpy_h)
print("--------")
print(torch_c)
print(numpy_c)

#############################################################

#使用pytorch的GRU层
torch_gru = nn.GRU(input_dim, hidden_size, batch_first=True)
# for key, weight in torch_gru.state_dict().items():
#     print(key, weight.shape)


#将pytorch的GRU网络权重拿出来，用numpy通过矩阵运算实现GRU的计算
def numpy_gru(x, state_dict):
    weight_ih = state_dict["weight_ih_l0"].numpy()
    weight_hh = state_dict["weight_hh_l0"].numpy()
    bias_ih = state_dict["bias_ih_l0"].numpy()
    bias_hh = state_dict["bias_hh_l0"].numpy()
    #pytorch将3个门的权重拼接存储，我们将它拆开
    w_r_x, w_z_x, w_x = weight_ih[0:hidden_size, :], \
                        weight_ih[hidden_size:hidden_size * 2, :],\
                        weight_ih[hidden_size * 2:hidden_size * 3, :]
    w_r_h, w_z_h, w_h = weight_hh[0:hidden_size, :], \
                        weight_hh[hidden_size:hidden_size * 2, :], \
                        weight_hh[hidden_size * 2:hidden_size * 3, :]
    b_r_x, b_z_x, b_x = bias_ih[0:hidden_size], \
                        bias_ih[hidden_size:hidden_size * 2], \
                        bias_ih[hidden_size * 2:hidden_size * 3]
    b_r_h, b_z_h, b_h = bias_hh[0:hidden_size], \
                        bias_hh[hidden_size:hidden_size * 2], \
                        bias_hh[hidden_size * 2:hidden_size * 3]
    w_z = np.concatenate([w_z_h, w_z_x], axis=1)
    w_r = np.concatenate([w_r_h, w_r_x], axis=1)
    b_z = b_z_h + b_z_x
    b_r = b_r_h + b_r_x
    h_t = np.zeros((1, hidden_size))
    sequence_output = []
    for x_t in x:
        x_t = x_t[np.newaxis, :]
        hx = np.concatenate([h_t, x_t], axis=1)
        z_t = sigmoid(np.dot(hx, w_z.T) + b_z)
        r_t = sigmoid(np.dot(hx, w_r.T) + b_r)
        h = np.tanh(r_t * (np.dot(h_t, w_h.T) + b_h) + np.dot(x_t, w_x.T) + b_x)
        h_t = (1 - z_t) * h + z_t * h_t
        sequence_output.append(h_t)
    return np.array(sequence_output), h_t

# torch_sequence_output, torch_h = torch_gru(torch.Tensor([x]))
# numpy_sequence_output, numpy_h = numpy_gru(x, torch_gru.state_dict())
#
# print(torch_sequence_output)
# print(numpy_sequence_output)
# print("--------")
# print(torch_h)
# print(numpy_h)

