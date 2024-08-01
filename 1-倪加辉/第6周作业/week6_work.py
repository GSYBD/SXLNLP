"""
week6-work :
计算Bert的参数量
"""
import torch

hidden_size = 768
num_attention_heads = 12
num_hidden_layers = 12
vocab_size = 21128
intermediate_size = 3072
#emb层
word_embeddings = vocab_size * hidden_size
segments_embeddings = 2 * hidden_size
positon_embeddings = 512 * hidden_size
# self-attention
Q_K_V_Weight= num_attention_heads * hidden_size
Q_K_V_Bias = num_attention_heads
attention_output_weight = hidden_size * hidden_size
attention_output_bias = hidden_size
# Norm 层
attention_layer_norm_weight = hidden_size
attention_layer_norm_bias = hidden_size
# 前向传播
intermediate_weight = hidden_size * intermediate_size
intermediate_bias = intermediate_size
# Norm 层
ff_layer_norm_weight = hidden_size
ff_layer_norm_bias = hidden_size

# pooler层
pooler_weight = hidden_size * hidden_size
pooler_bias = hidden_size

# 总参数量为  可能还需要×层数
total_params = word_embeddings + segments_embeddings + positon_embeddings + Q_K_V_Weight + Q_K_V_Bias + attention_output_weight + attention_output_bias + attention_layer_norm_weight + attention_layer_norm_bias + intermediate_weight + intermediate_bias + ff_layer_norm_weight + ff_layer_norm_bias + pooler_weight + pooler_bias
print(total_params)

