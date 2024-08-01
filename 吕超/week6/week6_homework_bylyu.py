# -*- encoding: utf-8 -*-
'''
week6_homework_bylyu.py
Created on 2024/7/25 20:46
@author: Allan Lyu
@Description: 计算bert中的可训练参数数量
'''

import math

import numpy as np
import torch
from transformers import BertModel

'''

通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models

'''

# bert = BertModel.from_pretrained(r"F:\Desktop\work_space\pretrain_models\bert-base-chinese", return_dict=False)
bert = BertModel.from_pretrained(r"D:\my_study\4_八斗AI\0_八斗精品班\6_第6周_预训练模型\bert-base-chinese",
                                 return_dict=False)  # lyu: 决定模型参数输出格式是否兼容新旧版本
# state_dict = bert.state_dict()
# bert.eval()
# x = np.array([2450, 15486, 102, 2110])  # 假想成4个字的句子, lyu :对应的值是在次表中的位置
# torch_x = torch.LongTensor([x])  # pytorch形式输入
# seqence_output, pooler_output = bert(torch_x)
# # lyu: 用法不一样,seqence_output是每个字对应的向量, 适用于每个字需要做分类场景; pooler_output首字符对应向量, 适用于每个句子做分类场景;
# print(seqence_output.shape, pooler_output.shape)  # torch.Size([1, 4, 768]) torch.Size([1, 768])
# # print(seqence_output, pooler_output)
#
# print(bert.state_dict().keys())  # 查看所有的权值矩阵名称


# # 方式1计算可训练参数
# total_params = 0
# for k, v in bert.state_dict().items():
#  # print(k, v.shape)
#  total_params += np.prod(v.shape)
#  print("Total trainable parameters: " , total_params)  # 24301056


# 方式2: 计算bert中的可训练参数数量 24301056
print("method2 : Total trainable parameters: ", sum(p.numel() for p in bert.parameters() if p.requires_grad))


# 方法3: 计算bert中的可训练参数数量
def calculate_bert_parameters(vocab_size, embedding_size, max_position_embeddings, num_hidden_layers, intermediate_size):
    # 嵌入层参数
    # 词嵌入 + 位置嵌入 + 句嵌入 + 嵌入层(归一化)
    embedding_params = (vocab_size + max_position_embeddings + 2) * embedding_size + embedding_size + embedding_size

    # 每个Transformer层的参数
    # 自注意力机制
    # Q, K, V 矩阵，每个都是 [embedding_size, num_heads * head_size]
    attention_params = 3 * (embedding_size * embedding_size + embedding_size)
    # 输出权重
    attention_output_params = embedding_size * embedding_size + embedding_size
    # transformer层(第1次归一化)（两个参数：gamma和beta）
    layer_norm_params = embedding_size + embedding_size

    # 前馈网络
    # 第一个线性层（假设使用ReLU激活）
    ffn_intermediate_params = embedding_size * intermediate_size + intermediate_size
    # 第二个线性层
    ffn_output_params = intermediate_size * embedding_size + embedding_size
    # transformer层(第2次归一化)（两个参数：gamma和beta）
    ffn_norm_params = embedding_size + embedding_size

    # 单个Transformer层的总参数
    single_transformer_params = attention_params + attention_output_params + layer_norm_params + ffn_intermediate_params + ffn_output_params + ffn_norm_params

    # 所有Transformer层的参数
    transformer_params = single_transformer_params * num_hidden_layers

    # 池化层
    # 注意：在标准的BERT中，池化层可能更简单，但这里我们按线性层计算
    pooler_params = embedding_size * embedding_size + embedding_size

    # 总参数
    total_params = embedding_params + transformer_params + pooler_params

    return total_params


vocab_size = 21128
embedding_size = 768
max_position_embeddings = 512
# num_attention_heads = 12
num_hidden_layers = 1
intermediate_size = 3072

# method3 : Total trainable parameters:  24889344
method3_total_params = calculate_bert_parameters(vocab_size, embedding_size, max_position_embeddings, num_hidden_layers,intermediate_size)
print("method3 : Total trainable parameters: ", method3_total_params)