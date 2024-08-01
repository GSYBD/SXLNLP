import torch
import math
import torch.nn as nn
import numpy as np
from transformers import BertModel

model = BertModel.from_pretrained(r"D:\BaiduNetdiskDownload\第六周 预训练模型\bert-base-chinese", return_dict=False)
n = 2                       # 输入最大句子个数
vocab = 21128               # 词表数目
max_sequence_length = 512   # 最大句子长度
embedding_size = 768        # embedding维度
hide_size = 3072            # 隐藏层维数


#首先x输入后进入embedding层，分为三部分，token embedding其参数是 词表vocab*embedding_size，这意味着对于词汇表中的每个单词，都有一个 embedding_size 维度的向量与之对应。
#segment embeddings其参数是embedding_size*embedding_size,用来表示不同句子，position embeddings其参数是用来表示每个序列的位置的词向量，max_sequence_length*embedding_size
# embedding_size + embedding_sizes是layer_norm层参数
embedding_quant=vocab*embedding_size+n*embedding_size+max_sequence_length*embedding_size+embedding_size+embedding_size
#self attention 自注意力机制中的权重矩阵是将嵌入向量转换为qkv而设置的向量矩阵权重，embedding_size*embedding_size，
self_attention=(embedding_size*embedding_size+embedding_size)*3
#self attention out 输出的权重参数是将，自注意力机制的向量参数进行一个线形层 embeddings_size*embeddings_size，以及其偏置embedding_size和embedding_size + embedding_size的layer_norm层参数
self_attention_out=embedding_size*embedding_size+embedding_size+embedding_size+embedding_size

#前向传播要经过两个线性层，embedding_size*hide_size+hide_size是第一个线性层，第一个线性层将其转换为hide_size形状的张量，第二个线性层将第一个线性层的矩阵转换为embedding_size形状的矩阵，其权重是
# hide_size*embedding_size+embedding_size这种矩阵形式，并且要加上后面的embedding_size + embedding_size是layer_norm层
feed_forward = embedding_size * hide_size + hide_size + embedding_size * hide_size + embedding_size + embedding_size + embedding_size

#结束之后进行池化操作，pool进行矩阵向量的池化操作，将向前传播的矩阵向量和池化层相乘
pool_quant=embedding_size*embedding_size+embedding_size

all_quant=embedding_quant+self_attention+self_attention_out+feed_forward+pool_quant

print("模型实际参数个数为%d" % sum(p.numel() for p in model.parameters()))
print("diy计算参数个数为%d" % all_quant)
