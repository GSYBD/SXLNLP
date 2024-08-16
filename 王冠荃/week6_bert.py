import torch
import math
import torch.nn as nn
import numpy as np
from transformers import BertModel

model = BertModel.from_pretrained(r"D:\NLP\第六周 预训练模型\bert-base-chinese", return_dict=False)

n = 2                       # 输入最大句子个数
vocab = 21128               # 词表数目
max_sequence_length = 512   # 最大句子长度
embedding_size = 768        # embedding维度
hide_size = 3072            # 隐藏层维数


print("Begin count!")

embedding_num = vocab*embedding_size + max_sequence_length*embedding_size + n*embedding_size + embedding_size + embedding_size

attention_num = 3*(embedding_size*embedding_size + embedding_size)

attention_output_num = embedding_size*embedding_size + embedding_size + embedding_size + embedding_size

feed_forward_num = (embedding_size*hide_size + hide_size) + (hide_size*embedding_size+embedding_size)
feed_forward_norm_num = embedding_size+embedding_size

pooler_num = embedding_size*embedding_size + embedding_size

total_num = embedding_num + attention_num + attention_output_num + feed_forward_num + feed_forward_norm_num + pooler_num

print("Total num:", total_num)

print("Real num:", sum(p.numel() for p in model.parameters()))


