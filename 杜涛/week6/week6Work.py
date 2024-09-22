import torch
import math
import numpy as np
from transformers import BertModel

'''
embedding层:
token embedding 权重   vocab_size * hidden_size 
segment embedding 权重 type_vocab_size * hidden_size
position embedding 权重 max_position_embeddings * hidden_size
layer normal归一化 权重: 
weight : hidden_size
bias : hidden_size
#transformer权重
q权重 weight,bias  hidden_size * hidden_size hidden_size
k权重 weight,bias  hidden_size * hidden_size hidden_size
v权重 weight,bias  hidden_size * hidden_size hidden_size
qkv :weight,bias  hidden_size * hidden_size hidden_size
线性层 :weight,bias  hidden_size * hidden_size hidden_size
layer normal: weight ,bias 权重 hidden_size, hidden_size
feed forward权重:
第一个线性层: weight,bias  (4 * hidden_size) * hidden_size  4 * hidden_size
第二个线性层: weight,bias  hidden_size * (4 * hidden_size)  hidden_size
layer normal归一化: weight ,bias 权重 hidden_size,hidden_size
pool层权重: weight,bias  hidden_size * hidden_size ， hidden_size
'''

# 计算bert的可训练参数数量
def get_bert_param_num():
    vocab_size = 21128
    hidden_size = 768
    max_position_embeddings = 512
    num_hidden_layers = 1  # 层数
    num_attention_heads = 12  # 多头机制只是将参数列拆分进行训练，不影响结果
    type_vocab_size = 2

    # embedding层
    token_embedding_weight = vocab_size * hidden_size
    type_embedding_weight = type_vocab_size * hidden_size
    position_embedding_weight = max_position_embeddings * hidden_size
    # layer normal归一化
    embeddings_layer_norm_weight = hidden_size
    embeddings_layer_norm_bias = hidden_size

    embedding_total = token_embedding_weight + type_embedding_weight + position_embedding_weight + embeddings_layer_norm_weight + embeddings_layer_norm_bias

    # transformer层
    # qkv
    q_w = hidden_size * hidden_size * num_hidden_layers
    q_b = hidden_size * num_hidden_layers
    k_w = hidden_size * hidden_size * num_hidden_layers
    k_b = hidden_size * num_hidden_layers
    v_w = hidden_size * hidden_size * num_hidden_layers
    v_b = hidden_size * num_hidden_layers

    size = embedding_total + q_w + q_b + k_w + k_b + v_w + v_b

    # 线性层
    attention_output_weight = hidden_size * hidden_size * num_hidden_layers
    attention_output_bias = hidden_size * num_hidden_layers
    # layer normal归一化+残差机制
    attention_layer_norm_w = hidden_size * num_hidden_layers
    attention_layer_norm_b = hidden_size * num_hidden_layers
    # feed forward层
    intermediate_size = 4 * hidden_size
    intermediate_weight = intermediate_size * hidden_size * num_hidden_layers
    intermediate_bias = intermediate_size * num_hidden_layers

    output_weight = hidden_size * intermediate_size * num_hidden_layers
    output_bias = hidden_size * num_hidden_layers

    # layer normal归一化+残差机制
    ff_layer_norm_w = hidden_size * num_hidden_layers
    ff_layer_norm_b = hidden_size * num_hidden_layers

    # pool层
    pooler_dense_weight = hidden_size * hidden_size
    pooler_dense_bias = hidden_size

    size = size + attention_output_weight + attention_output_bias + \
           attention_layer_norm_w + attention_layer_norm_b + intermediate_weight + intermediate_bias + \
           output_weight + output_bias + ff_layer_norm_w + ff_layer_norm_b + pooler_dense_weight + pooler_dense_bias

    return size

if __name__ == '__main__':
    size = get_bert_param_num()
    print("bert的可训练参数数量: %d" % size)
