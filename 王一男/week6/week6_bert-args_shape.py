"""
week6-work :
计算Bert的参数量
"""
import torch
import numpy as np

sentence_len = 512
batches = 1000  # 不影响参数量不重要）
input_size = (batches, sentence_len)
vocab_len = 21128
embedding_dim = 768
hidden_dim = 768
layer_num = 12
# embedding_shape = (vocab_len, embedding_dim)
x = (batches, sentence_len, embedding_dim)
embedding_shape = vocab_len * embedding_dim
embedding_args = embedding_shape + sentence_len * embedding_dim + 2 * embedding_dim
print(embedding_args)  # embedding层参数量
# we+pe+te 然后过一个归一化层?
norm_args = 2 * embedding_dim
args = 0
# 然后进入transformer结构，使用multi_head self_attention
transformer_args = 0
multi_heads = 12  # 多头结构并不影响参数量，只是将参数列拆分分别训练
x_shape = (batches, sentence_len, embedding_dim)
dim_qk = hidden_dim
Q_w_shape = (embedding_dim, dim_qk)
Q_b_shape = dim_qk
Q_shape = (batches, sentence_len, dim_qk)
seqlen = sentence_len
embedding_dim2 = embedding_dim
x2_shape = (batches, seqlen, embedding_dim2)
K_w_shape = (embedding_dim2, dim_qk)  # np.random.rand()
K_b_shape = dim_qk
K_shape = (batches, seqlen, dim_qk)
QK_shape = (batches, sentence_len, seqlen)
dim_v = hidden_dim
V_w_shape = (seqlen, dim_v)
V_b_shape = dim_v
V_shape = (batches, sentence_len, dim_v)
self_attention_args = 3 * embedding_dim * hidden_dim + 3 * hidden_dim
# 因为需要叠加多层，所以hidden_dim==embedding_dim 便于多次注入
# 多头注意力的分配权重dense层，linear结构
attention_output_dense_args = hidden_dim ** 2 + hidden_dim
# 残差网络+归一化
# args += norm_args
transformer_args += (self_attention_args + norm_args + attention_output_dense_args)
# FFN层+残差&归一化(Add&Norm)
intermediate_dim = 4 * hidden_dim
FFN_args = (hidden_dim * intermediate_dim + intermediate_dim
            + intermediate_dim * embedding_dim + embedding_dim)
transformer_args += (FFN_args + norm_args)
args = embedding_args + layer_num * transformer_args
print(self_attention_args, attention_output_dense_args, FFN_args, norm_args, transformer_args)
# 最后的pooler层
pooler_dense_args = hidden_dim * hidden_dim + hidden_dim
args = embedding_args + layer_num * transformer_args + pooler_dense_args
print(args)


