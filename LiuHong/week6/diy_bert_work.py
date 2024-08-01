import torch
import math
import numpy as np
from transformers import BertModel

'''

通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models

'''

bert = BertModel.from_pretrained(r"bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()

for k,v in state_dict.items():
    print(k,v.shape)

total_params = sum(p.numel() for p in bert.parameters())
print(f'bert nums {total_params}')
# 102267648



parameter = bert.config
# Embedding参数:
hidden_size = parameter.hidden_size
embeddings_vocab = parameter.vocab_size * hidden_size
embeddings_position = parameter.max_position_embeddings * hidden_size
embeddings_segment = parameter.type_vocab_size * hidden_size
embeddings_ln = hidden_size + hidden_size
#Embedding层的总参数量
embeddings_nums = embeddings_vocab + embeddings_position + embeddings_segment + embeddings_ln
# Transformer参数:
# Q k v 多头
attention_qkv = 3 * (hidden_size * hidden_size + hidden_size)
densen_sum = hidden_size * hidden_size + hidden_size
attetion_ln = 2 * hidden_size

self_attention_nums = attention_qkv  + densen_sum + attetion_ln
#feed_forwar:
feedforward_intermediate1 = hidden_size * parameter.intermediate_size + parameter.intermediate_size
feedforward_intermediate2 = parameter.intermediate_size * hidden_size + hidden_size
feedforward_ln = 2 * hidden_size
feedforward_nums = feedforward_intermediate1 + feedforward_intermediate2  + feedforward_ln
# ----------Transformer总参数:----------
transformer_nums =  (self_attention_nums + feedforward_nums) * parameter.num_hidden_layers
# pooling层
pooling_layer = hidden_size * hidden_size
pooling_bias = hidden_size
#----------pooling层总参数----------
pooling_nums = pooling_layer + pooling_bias
# bert总参数量 102267648
bert_nums = embeddings_nums + transformer_nums + pooling_nums
print('self bert_nums:', bert_nums)


