import torch
import math
import numpy as np
from transformers import BertModel
import os
import json

'''

通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models

'''

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 通过config.json计算bert模型权限数量
with open(r'..\bert-base-chinese\config.json') as f:
    # 使用 read() 方法来获取文件内容
    content = f.read()
    # 使用 json.loads() 将 JSON 字符串转换为 Python 对象
    data = json.loads(content)
# print(data)

# embedding层权重数量
# embedding输入层
token_embedding_weight = data['vocab_size'] * data['hidden_size']
position_embedding_weight = data['max_position_embeddings'] * data['hidden_size']
segment_embedding_weight = data['type_vocab_size'] * data['hidden_size']
embedding_layer_norm_w = data['hidden_size'] * 2
total = token_embedding_weight + position_embedding_weight + segment_embedding_weight + embedding_layer_norm_w

# 计算encoder层的权重数量
# self-attention
q_w = data['hidden_size'] * data['hidden_size']
k_w = data['hidden_size'] * data['hidden_size']
v_w = data['hidden_size'] * data['hidden_size']
q_b = data['hidden_size']
k_b = data['hidden_size']
v_b = data['hidden_size']
# output self-attention
attention_output_weight = data['hidden_size'] * data['hidden_size']
attention_output_bias = data['hidden_size']
# layer norm self-attention 
attention_layer_norm_w = data['hidden_size'] * 2
# liner intermediate feedback
intermediate_weight = data['intermediate_size'] * data['hidden_size']
intermediate_bias = data['intermediate_size']
# liner output feedback 
output_weight = data['hidden_size'] * data['intermediate_size']
output_bias = data['hidden_size']
# layer norm feedback
ff_layer_norm_w = data['hidden_size'] * 2
# 累加encoder层权重
total += q_w + k_w + v_w + q_b + k_b + v_b + attention_output_weight + attention_output_bias + attention_layer_norm_w + intermediate_weight + intermediate_bias + output_weight + output_bias + ff_layer_norm_w
# 乘隐含层数
total *= data['num_hidden_layers']
print("总权重数量：", total)

# bert = BertModel.from_pretrained(r"D:\移动云盘同步盘\其他\AI\资料\week6 语言模型和预训练\bert-base-chinese",from_tf=True, return_dict=False)
# state_dict = bert.state_dict()
# num_hidden_layers = 1  # 定义bert中的encoder的层数，当前加载模型的层数=1 

# # 计算embedding层的权重数量
# word_embeddings_shape = state_dict["embeddings.word_embeddings.weight"].reshape(-1).shape
# token_type_embeddings_shape = state_dict["embeddings.token_type_embeddings.weight"].reshape(-1).shape
# position_embeddings_shape = state_dict["embeddings.position_embeddings.weight"].reshape(-1).shape
# layernorm_embeddings_weight_shape = state_dict["embeddings.LayerNorm.weight"].reshape(-1).shape
# layernorm_embeddings_bias_shape = state_dict["embeddings.LayerNorm.bias"].reshape(-1).shape
# # 累加以上权重
# total = word_embeddings_shape + token_type_embeddings_shape + position_embeddings_shape + layernorm_embeddings_weight_shape + layernorm_embeddings_bias_shape


# # 计算encoder层的权重数量
# q_w = state_dict["encoder.layer.0.attention.self.query.weight"].reshape(-1).shape
# k_w = state_dict["encoder.layer.0.attention.self.key.weight"].reshape(-1).shape
# v_w = state_dict["encoder.layer.0.attention.self.value.weight"].reshape(-1).shape
# q_b = state_dict["encoder.layer.0.attention.self.query.bias"].reshape(-1).shape
# k_b = state_dict["encoder.layer.0.attention.self.key.bias"].reshape(-1).shape
# v_b = state_dict["encoder.layer.0.attention.self.value.bias"].reshape(-1).shape
# attention_output_weight = state_dict["encoder.layer.0.attention.output.dense.weight"].reshape(-1).shape
# attention_output_bias = state_dict["encoder.layer.0.attention.output.dense.bias"].reshape(-1).shape
# attention_layer_norm_w = state_dict["encoder.layer.0.attention.output.LayerNorm.weight"].reshape(-1).shape
# attention_layer_norm_b = state_dict["encoder.layer.0.attention.output.LayerNorm.bias"].reshape(-1).shape
# intermediate_weight = state_dict["encoder.layer.0.intermediate.dense.weight"].reshape(-1).shape
# intermediate_bias = state_dict["encoder.layer.0.intermediate.dense.bias"].reshape(-1).shape
# output_weight = state_dict["encoder.layer.0.output.dense.weight" ].reshape(-1).shape
# output_bias = state_dict["encoder.layer.0.output.dense.bias"].reshape(-1).shape
# ff_layer_norm_w = state_dict["encoder.layer.0.output.LayerNorm.weight"].reshape(-1).shape
# ff_layer_norm_b = state_dict["encoder.layer.0.output.LayerNorm.bias"].reshape(-1).shape
# # 累加以上权重
# total += q_w + k_w + v_w + q_b + k_b + v_b + attention_output_weight + attention_output_bias + attention_layer_norm_w + attention_layer_norm_b + intermediate_weight + intermediate_bias + output_weight + output_bias + ff_layer_norm_w + ff_layer_norm_b
# # 乘隐含层数
# total *= num_hidden_layers

# # pooler层权重
# pooler_dense_weight = state_dict["pooler.dense.weight"].reshape(-1).shape
# pooler_dense_bias = state_dict["pooler.dense.bias"].reshape(-1).shape
# total += pooler_dense_bias + pooler_dense_weight

# print("总权重数量：", sum(total))  
# print(np.prod(state_dict.items()))


# print("============================================================")
# print("embedding层的可训练参数计算")
# # 从state_dict中获取embeding层的输入参数，vocab_size & hidden_size = torch.Size([21128, 768])
# word_embeddings_vocab_size = state_dict["embeddings.word_embeddings.weight"].shape[0]
# word_embeddings_hidden_size = state_dict["embeddings.word_embeddings.weight"].shape[1]
# # 计算embeding层的可训练参数
# print("word_embedding层可训练参数(word_embedings_vocab_size * word_embedings_hidden_size)：", word_embeddings_vocab_size * word_embeddings_hidden_size)


# # torch.Size([2, 768])
# token_type_embeddings_shape = state_dict["embeddings.token_type_embeddings.weight"].shape
# token_type_embeddings_sentence_len = token_type_embeddings_shape[0]
# token_type_embeddings_hidden_size = token_type_embeddings_shape[1]
# print("token_type_embeddings层可训练参数(token_type_embeddings_sentence_len*token_type_embeddings_hidden_size)：", token_type_embeddings_sentence_len*token_type_embeddings_hidden_size)
# # print(token_type_embeddings_shape)

# # torch.Size([512, 768])
# position_embeddings_shape = state_dict["embeddings.position_embeddings.weight"].shape
# position_embeddings_sentence_len = position_embeddings_shape[0]
# position_embeddings_hidden_size = position_embeddings_shape[1]
# print("position_embeddings层可训练参数(position_embeddings_sentence_len*position_embeddings_hidden_size)：", position_embeddings_sentence_len*position_embeddings_hidden_size)
# # print(position_embeddings_shape)

# # torch.Size([768])
# layernorm_embeddings_weight_shape = state_dict["embeddings.LayerNorm.weight"].shape
# layernorm_embeddings_weight_shape = layernorm_embeddings_weight_shape[0]
# print("layernorm_embeddings层可训练参数(layernorm_embeddings_weight_shape)：", layernorm_embeddings_weight_shape)
# # print(layernorm_embeddings_shape)

# # torch.Size([768])
# layernorm_embeddings_bias_shape = state_dict["embeddings.LayerNorm.bias"].shape
# layernorm_embeddings_bias_size = layernorm_embeddings_bias_shape[0]
# print("layernorm_embeddings层可训练参数(layernorm_embeddings_bias_size)：", layernorm_embeddings_bias_size)
# print(layernorm_embeddings_bias_shape)

# # bert-encoder部分，有多层，以第一层为例
# # bert self-attention
# # wq torch.Size([768, 768])
# q_w = state_dict["encoder.layer.0.attention.self.query.weight"]
# print("q_w.shape", q_w.shape)
# # wk torch.Size([768, 768])
# k_w = state_dict["encoder.layer.0.attention.self.key.weight"]
# print("k_w.shape", k_w.shape)
# # wv torch.Size([768, 768])
# v_w = state_dict["encoder.layer.0.attention.self.value.weight"]
# print("v_w.shape", v_w.shape)
# # 计算q_w,k_w,v_w的可训练参数
# print("q_w,k_w,v_w的可训练参数(q_w + k_w+ v_w)：", q_w.shape[0]*q_w.shape[1] + k_w.shape[0]*k_w.shape[1]+ v_w.shape[0]*v_w.shape[1])

# # bq torch.Size([768])
# q_b = state_dict["encoder.layer.0.attention.self.query.bias"]
# print("q_w.shape", q_b.shape)
# # bk torch.Size([768])
# k_b = state_dict["encoder.layer.0.attention.self.key.bias"]
# print("k_w.shape", q_b.shape)
# # bv torch.Size([768])
# v_b = state_dict["encoder.layer.0.attention.self.value.bias"]
# print("v_w.shape", q_b.shape)


# # Liner(Attention(Q,K,V)) 
# # weight torch.Size([768, 768])
# attention_output_weight = state_dict["encoder.layer.0.attention.output.dense.weight"]
# print("attention_output_weight.shape", attention_output_weight.reshape(-1).shape)
# # bias torch.Size([768])
# attention_output_bias = state_dict["encoder.layer.0.attention.output.dense.bias"]
# print("attention_output_bias.shape", attention_output_bias.shape)

# # LayerNorm(Xembedding+ Xattention)
# # torch.Size([768])
# attention_layer_norm_w = state_dict["encoder.layer.0.attention.output.LayerNorm.weight"]
# # torch.Size([768])
# attention_layer_norm_b = state_dict["encoder.layer.0.attention.output.LayerNorm.bias"]
# print("attention_layer_norm_w.shape", attention_layer_norm_w.shape)
# print("attention_layer_norm_b.shape", attention_layer_norm_b.shape)

# # Liner(x)
# # liner hidden_size * 4 : from torch.Size([768, 768]) to torch.Size([3072, 768])
# intermediate_weight = state_dict["encoder.layer.0.intermediate.dense.weight"]
# # torch.Size([3072])
# intermediate_bias = state_dict["encoder.layer.0.intermediate.dense.bias"]
# print ("intermediate_weight.shape", intermediate_weight.shape)
# print ("intermediate_bias.shape", intermediate_bias.shape)

# # Liner(gelu(Liner(x)))
# # torch.Size([768, 3072])
# output_weight = state_dict["encoder.layer.0.output.dense.weight" ]
# # torch.Size([768])
# output_bias = state_dict["encoder.layer.0.output.dense.bias"]
# print ("output_weight.shape", output_weight.shape)
# print ("output_bias.shape", output_bias.shape)

# # LayerNorm(Xembedding+ Xattention) 
# # torch.Size([768])
# ff_layer_norm_w = state_dict["encoder.layer.0.output.LayerNorm.weight"]
# # torch.Size([768])
# ff_layer_norm_b = state_dict["encoder.layer.0.output.LayerNorm.bias"]
# print("ff_layer_norm_w.shape", ff_layer_norm_w.shape)
# print("ff_layer_norm_b.shape", ff_layer_norm_b.shape)


# #pooler层
# pooler_dense_weight = state_dict["pooler.dense.weight"]
# print("pooler_dense_weight.shape", pooler_dense_weight.shape)   
# pooler_dense_bias = state_dict["pooler.dense.bias"]
# print("pooler_dense_bias.shape", pooler_dense_bias.shape)
# print("============================================================")
