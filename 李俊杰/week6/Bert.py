import torch
import torch.nn as nn
import numpy as np
import random
import json
from transformers import BertModel

attention_probs_dropout_prob = 0.1
directionality = "bidi"
hidden_act = "gelu"
hidden_dropout_prob = 0.1
hidden_size = 768
initializer_range = 0.02
intermediate_size = 3072
layer_norm_eps = 1e-12
max_position_embeddings = 512
model_type = "bert"
num_attention_heads = 12
num_hidden_layers = 1
pad_token_id = 0
pooler_fc_size = 768
pooler_num_attention_heads = 12
pooler_num_fc_layers = 3
pooler_size_per_head = 128
pooler_type = "first_token_transform"
type_vocab_size = 2
vocab_size = 21128
num_labels = 18

# Embedding： 词表大小*隐单元，语序*隐单元，最大输入*隐单元
embedding_sum = (vocab_size + type_vocab_size + max_position_embeddings) * hidden_size
# 归一化（输入减均值除方差）：w，b
layer_norm_sum = hidden_size * 2
print(f"Embedding层参数={embedding_sum + layer_norm_sum}")

# 注意力机制：Q，K，V。w，b
attention_sum = 3 * (hidden_size * hidden_size + hidden_size)
# 残差快+归一化
densen1_sum = (hidden_size * hidden_size + hidden_size) + layer_norm_sum
print(f"Attention层参数={attention_sum + densen1_sum}")

# feed_forwar,一个线性层过激活函数后再过一个线性层
forward_sum = 2 * (intermediate_size * hidden_size + intermediate_size)
# 残差快+归一化
densen2_sum = (hidden_size * intermediate_size + hidden_size) + layer_norm_sum
print(f"Feed_Forwar层参数={forward_sum + densen2_sum}")

# 池化层
pooler_sum = hidden_size * hidden_size + hidden_size
# 总和=embedding层 + 12个transformer层 + 池化层
sum = (embedding_sum + layer_norm_sum) + pooler_num_attention_heads * (
        attention_sum + densen1_sum + forward_sum + densen2_sum) + pooler_sum

# Embedding层参数=16622592
# Attention层参数=2363904
# Feed_Forwar层参数=7086336
# 130616064
print(sum)  # 130616064
