import torch
import math
import numpy as np
from transformers import BertModel
import json
'''

通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models

'''

# download bert-base-chinese
bert = BertModel.from_pretrained("./bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()

dict_path = "./bert-base-chinese/config.json"
dict = json.load(open(dict_path, 'r'))
print(dict)

# Token Embeddings
token_param = dict['vocab_size'] * dict['hidden_size']
print(f"Token Embeddings: {token_param}")

# Segment Embeddings
segment_param = 2 * dict['hidden_size']
print(f"Segment Embeddings: {segment_param}")

# Position Embeddings
max_position_embeddings = dict['max_position_embeddings'] * dict['hidden_size']
print(f"Position Embeddings: {max_position_embeddings}")

# LayerNorm Embeddings
ln_param = dict['hidden_size']
print(f"LayerNorm Embeddings: Weights: {ln_param}")

# Attention Embeddings
# Q, K, V (multi-head no affect)
attention_param = dict['hidden_size'] * dict['hidden_size']
print(f"Attention Embeddings: for Q, K, V weights: {attention_param}")

# Linear before Gelu
linear_param = dict['hidden_size'] * dict['intermediate_size']
print(f"Linear before Gelu: {linear_param}")

# Linear after Gelu
linear_param = dict['intermediate_size'] * dict['hidden_size']
print(f"Linear after Gelu: {linear_param}")

# LayerNorm after Linear
ln_param = dict['hidden_size']
print(f"LayerNorm after Linear: Weights: {ln_param}")

# Output Layer
output_param = dict['hidden_size'] * dict['hidden_size']
print(f"Output Layer: {output_param}")


# for k, v in state_dict.items():
#     print(k, v.shape)