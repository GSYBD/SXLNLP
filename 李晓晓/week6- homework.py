import torch
import math
import numpy as np
from transformers import BertModel
'''
计算Bert中可训练参数数量
模型文件下载 https://huggingface.co/models
'''


def count_parameters(vocab_size, hidden_size, max_position_embeddings, num_attention_heads, attention_head_size,
                     intermediate_size, num_layers):
    # 嵌入层
    embedding_params = vocab_size * hidden_size + max_position_embeddings * hidden_size

    # Transformer层
    transformer_params = 0
    for _ in range(num_layers):
        # 自注意力机制
        attention_params = 3 * hidden_size * attention_head_size  # query, key, value
        attention_output_params = hidden_size * hidden_size + hidden_size  # weight and bias

        # 前馈网络
        ffn_params = hidden_size * intermediate_size + intermediate_size * hidden_size + hidden_size  # two weights and one bias

        transformer_params += attention_params + attention_output_params + ffn_params

        # 输出层
    output_params = hidden_size * hidden_size + hidden_size  # pooler_output_layer

    total_params = embedding_params + transformer_params + output_params
    return total_params


# 示例参数
vocab_size = 30522  # BERT-Base的参数
hidden_size = 768
max_position_embeddings = 512
num_attention_heads = 12
attention_head_size = hidden_size // num_attention_heads
intermediate_size = hidden_size * 4
num_layers = 12

# 计算参数数量
total_params = count_parameters(vocab_size, hidden_size, max_position_embeddings, num_attention_heads,
                                attention_head_size, intermediate_size, num_layers)
print(f"Total trainable parameters: {total_params}")