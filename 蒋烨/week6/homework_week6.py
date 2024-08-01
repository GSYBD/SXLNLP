import torch
import math
import numpy as np
from transformers import BertModel


# bert = BertModel.from_pretrained("bert-base-chinese", return_dict=False)
# state_dict = bert.state_dict()
#
# # v.shape可以计算bert里面的可训练参数
# for k, v in state_dict.items():
#     print(k, v.shape)


vocab_size = 21128
type_vocab_size = 2
hidden_size = 768
max_position_embedding = 512
intermediate_size = 3072
num_hidden_layers = 1

# 1.Embedding层
# token+segment+position+layer normalization
embedding = vocab_size * hidden_size + type_vocab_size * hidden_size + max_position_embedding * hidden_size + hidden_size + hidden_size

# 2.Self-attention
# Q,K,V3个线性层+Linear
attention = hidden_size * hidden_size * 3 + hidden_size * 3 + hidden_size * hidden_size + hidden_size

# 3.LayerNorm
# embedding+attention
layernorm = (hidden_size + hidden_size) * 2

# 4.Feed Forward
forward = (hidden_size * intermediate_size + intermediate_size) + (intermediate_size * hidden_size + hidden_size)

# 5.Pooler层
pooler = hidden_size*hidden_size+hidden_size

result = embedding + attention + layernorm + forward + pooler
print(result)
