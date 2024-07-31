import torch
import math
import numpy as np
from transformers import BertModel

'''

通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models

'''

bert = BertModel.from_pretrained(r"E:\badouai\ai\week6\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
# diy_result
hidden_size = 768
max_position_embeddings=512
num_hidden_layers=1
type_vocab_size=2
vocab_size=21128

parameters_embedding = vocab_size * hidden_size + max_position_embeddings * hidden_size + type_vocab_size * hidden_size + hidden_size + hidden_size
parameters_attention = (hidden_size * hidden_size + hidden_size) * 3
parameters_output1 = hidden_size * hidden_size + hidden_size
parameters_laterNorm1 = hidden_size + hidden_size
parameters_output2 = (hidden_size * hidden_size * 4 + hidden_size * 4) + (hidden_size * 4 * hidden_size + hidden_size)
parameters_laterNorm2 = hidden_size + hidden_size
parameters_pooler = (hidden_size * hidden_size + hidden_size)
total_number = num_hidden_layers * (parameters_embedding + parameters_attention + parameters_output1 + parameters_laterNorm1 + parameters_output2 + parameters_laterNorm2 + parameters_pooler)
print("diy result",total_number)

# check result
count = 0
for k, v in state_dict.items():
    if len(v.shape) == 1:
        count += v.shape[0]
    elif len(v.shape) == 2:
        count += v.shape[0] * v.shape[1]
print("model result",count)
