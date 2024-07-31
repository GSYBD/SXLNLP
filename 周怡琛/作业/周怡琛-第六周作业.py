# Define the parameters
V = 30522  # 词汇表大小
H = 768    # 隐藏
L = 12     # 层数
A = 12     # 注意力层数

embedding_params = V * H
attention_params_per_layer = A * 3 * H * H + H * H
ffn_params_per_layer = 8 * H * H
layer_norm_params_per_layer = 4 * H

total_params = embedding_params + L * (attention_params_per_layer + ffn_params_per_layer + layer_norm_params_per_layer)

total_params