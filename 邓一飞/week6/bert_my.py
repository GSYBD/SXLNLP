
import torch
import math
import numpy as np
from transformers import BertModel

bert = BertModel.from_pretrained(r"D:\aiproject\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
#print(state_dict)
# for k, v in state_dict.items():
#     print(k)
#     print(v.shape)
#     print('----------')
cc = 0
print('embeddings参数>>>')
print(bert.embeddings.word_embeddings.weight.shape)
cc+=21128*768
print(bert.embeddings.position_embeddings.weight.shape)
cc+=512*768
print(bert.embeddings.token_type_embeddings.weight.shape)
cc+=2*768
cc+=768 #LN 参数
print('词向量训练参数:%s'%(cc))


print('encoder,Multi-head Attention 参数>>>')
num_hidden_layers=12
#Multi-head Attention*3*12 + 。。。
multi_params = (768*64*3*12+768*768+768+768*3072+3072*768+768)*num_hidden_layers
print(multi_params)

# print('FeedForward 参数')
# feedForward_params =(768*3072+3072*768)*num_hidden_layers

print('Pooler 参数>>>')
pooler_params = 768*768
print(pooler_params)

print('总参数量:%s'%(cc+multi_params+pooler_params))


# bert.eval()
# x = np.array([[2450, 15486, 102, 2110]])   #假想成4个字的句子
# torch_x = torch.LongTensor(x)          #pytorch形式输入
# seqence_output, pooler_output = bert(torch_x)
# print(seqence_output.shape, pooler_output.shape)