import torch
import math
import torch.nn as nn
import numpy as np
from transformers import BertModel

model = BertModel.from_pretrained(r"D:\ai预习课件\week6 语言模型和预训练\bert-base-chinese", return_dict=False)
n = 2                       # 输入最大句子个数
vocab = 21128               # 词表数目
max_sequence_length = 512   # 最大句子长度
embedding_size = 768        # embedding维度
hide_size = 3072            # 隐藏层维数

#权重形状

# embedding过程中的参数：
# vocab * embedding_size是Token embedding的参数， v*768
# max_sequence_length * embedding_size是Position embedding的参数，最大：512*768
# n * embedding_size是segment embedding的参数， 2*768
# embedding_size + embedding_size是layer_norm层参数
embedding_parameters = vocab * embedding_size + max_sequence_length * embedding_size + n * embedding_size + embedding_size + embedding_size

# self_attention过程的参数：
# embedding_size * embedding_size是每个变换权重矩阵w参数，例如qw
# embedding_size是每个变换的bias偏置b参数，例如qb
# *3是K Q V三个参与变换的总数目
self_attention_parameters = (embedding_size * embedding_size + embedding_size) * 3

# self_attention_out参数：
# 自注意力输出部分：
# embedding_size * embedding_size + embedding_size 是self输出的线性层wx+b参数（w矩阵参数+b偏置参数）
# embedding_size + embedding_size是归一化layer_norm层参数：缩放参数Gamma大小是embedding_size，偏置参数也是embedding_size
# 归一化layer_norm层 w的目的是对于原始数据的放缩，一般情况下是变小，变小的时候整体数值有偏移，所以加一个可训练的w来控制这种偏移，跟线性层不一样。
self_attention_out_parameters = embedding_size * embedding_size + embedding_size + embedding_size + embedding_size

# Feed Forward参数：
# embedding_size * hide_size + hide_size第一个线性层，偏置项作用是为每个隐藏单元提供一个偏移量，
# 为什么偏置项是hide_size，因为如公式所示，线性层会先映射成 4倍的hide_size，再映射回embedding_size
# 所以为了进行同纬度的相加，偏置项是hide_size
# embedding_size * hide_size + embedding_size第二个线性层，偏置项作用是为了保持输出维度与输入维度（embedding_size）的一致性
# embedding_size + embedding_size是layer_norm层
feed_forward_parameters = embedding_size * hide_size + hide_size + embedding_size * hide_size + embedding_size + embedding_size + embedding_size

# pool_fc层参数
pool_fc_parameters = embedding_size * embedding_size + embedding_size

# 模型总参数 = embedding层参数 + self_attention参数 + self_attention_out参数 + Feed_Forward参数 + pool_fc层参数
all_paramerters = embedding_parameters + self_attention_parameters + self_attention_out_parameters + \
    feed_forward_parameters + pool_fc_parameters
print("模型实际参数个数为%d" % sum(p.numel() for p in model.parameters()))
print("diy计算参数个数为%d" % all_paramerters)