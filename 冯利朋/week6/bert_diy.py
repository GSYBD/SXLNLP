import random
import sys

import numpy as np
from transformers import BertModel
import math
import torch
bert = BertModel.from_pretrained("/Users/gonghengan/Documents/hugging-face/bert-base-chinese", return_dict=False)
#
bert_static_dict = bert.state_dict()
# print(bert_static_dict)
x = np.array([2450, 15486, 15167, 2110])
torch_x = torch.LongTensor([x])
#softmax归一化
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

#gelu激活函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

# 归一化函数
def layer_norm(x, w, b):
    x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
    x = x * w + b
    return x
class DiyModel:
    def __init__(self, state_dict):
        self.state_dict = state_dict
        # 头数，多头机制
        self.num_attention_head = 12
        # 维度
        self.hidden_size = 768
        # 几层transformer
        self.num_layers = 12
        # 加载权重
        self.load_weight(self.state_dict)
    def load_weight(self,state_dict):
        # embedding层权重
        self.word_embeddings = state_dict['embeddings.word_embeddings.weight'].numpy()  # token embedding 随机初始化层
        self.position_embeddings = state_dict['embeddings.position_embeddings.weight'].numpy()  # position embedding # 位置信息，语序信息
        self.token_type_embeddings = state_dict['embeddings.token_type_embeddings.weight'].numpy()  # 句子顺序embedding, 两个向量
        # embedding层加完完毕，归一化的权重
        self.embeddings_layer_norm_weight = state_dict['embeddings.LayerNorm.weight'].numpy()
        self.embeddings_layer_norm_bias = state_dict['embeddings.LayerNorm.bias'].numpy()

        # 可能有多个transformer层，记录每个trans层的权重
        self.transformer_weights=[]
        for i in range(self.num_layers):
            # 所有的q层权重
            q_w = state_dict[f'encoder.layer.{i}.attention.self.query.weight'].numpy()
            q_b = state_dict[f'encoder.layer.{i}.attention.self.query.bias'].numpy()
            # 所有的k层权重
            k_w = state_dict[f'encoder.layer.{i}.attention.self.key.weight'].numpy()
            k_b = state_dict[f'encoder.layer.{i}.attention.self.key.bias'].numpy()
            # 所有的v层权重
            v_w = state_dict[f'encoder.layer.{i}.attention.self.value.weight'].numpy()
            v_b = state_dict[f'encoder.layer.{i}.attention.self.value.bias'].numpy()

            # attention_output_weight 执行self-attention计算以后，会经历一个线性层计算
            attention_output_weight = state_dict[f'encoder.layer.{i}.attention.output.dense.weight'].numpy()
            attention_output_bias = state_dict[f'encoder.layer.{i}.attention.output.dense.bias'].numpy()

            # output.LayerNorm.weight  执行self-attention 以后进行layer_norm的权重
            attention_layer_norm_w = state_dict[f'encoder.layer.{i}.attention.output.LayerNorm.weight'].numpy()
            attention_layer_norm_b = state_dict[f'encoder.layer.{i}.attention.output.LayerNorm.bias'].numpy()
            # intermediate_weight feed-forward层的权重
            intermediate_weight = state_dict[f'encoder.layer.{i}.intermediate.dense.weight'].numpy()
            intermediate_bias = state_dict[f'encoder.layer.{i}.intermediate.dense.bias'].numpy()

            # output_weight feed-forward层的权重
            output_weight = state_dict[f'encoder.layer.{i}.output.dense.weight'].numpy()
            output_bias = state_dict[f'encoder.layer.{i}.output.dense.bias'].numpy()

            # ff_layer_norm_w feed-forward的归一化层的权重
            ff_layer_norm_w = state_dict[f'encoder.layer.{i}.output.LayerNorm.weight'].numpy()
            ff_layer_norm_b = state_dict[f'encoder.layer.{i}.output.LayerNorm.bias'].numpy()
            # 存储
            self.transformer_weights.append(
                [q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias, attention_layer_norm_w,
                 attention_layer_norm_b,
                 intermediate_weight, intermediate_bias, output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
        # pool层权重
        self.pooler_dense_weight = state_dict['pooler.dense.weight'].numpy()
        self.pooler_dense_bias = state_dict['pooler.dense.bias'].numpy()

    # embedd层计算
    def get_embedding(self, embedding_weight,x):
        return np.array([embedding_weight[x] for x in x])
    def embedding_forward(self, x):
        te = self.get_embedding(self.word_embeddings, x)
        se = self.get_embedding(self.token_type_embeddings,np.array([0] * len(x)))
        pe = self.get_embedding(self.position_embeddings, np.array(list(range(0, len(x)))))
        embedding_context = te + se + pe
        return layer_norm(embedding_context,self.embeddings_layer_norm_weight,self.embeddings_layer_norm_bias)

    # transformer层计算
    def all_transformer(self, x):
        for i in range(0,self.num_layers):
            x = self.single_transformer(x, i)
        return x
    # 计算一层的transformer
    def single_transformer(self,x, i):
        q_w, q_b, k_w, k_b, v_w, v_b,\
        attention_output_weight, attention_output_bias,\
        attention_layer_norm_w,attention_layer_norm_b,\
        intermediate_weight, intermediate_bias, \
        output_weight, output_bias, \
        ff_layer_norm_w, ff_layer_norm_b = self.transformer_weights[i]

        attention_output = self.self_attention(attention_output_bias, attention_output_weight, k_b, k_w, q_b, q_w, v_b, v_w, x)

        x = layer_norm(x + attention_output,attention_layer_norm_w,attention_layer_norm_b)

        # feed_forward
        feed_x = self.feed_forward(x,intermediate_weight, intermediate_bias,output_weight, output_bias)

        x = layer_norm(x + feed_x,ff_layer_norm_w,ff_layer_norm_b)
        return x


    def self_attention(self, attention_output_bias, attention_output_weight, k_b, k_w, q_b, q_w, v_b, v_w, x):
        # 计算q,k,v
        q = np.dot(x, q_w.T) + q_b
        k = np.dot(x, k_w.T) + k_b
        v = np.dot(x, v_w.T) + v_b
        attention_size = int(self.hidden_size / self.num_layers)
        q = self.transformer_score(q, self.num_attention_head, attention_size)
        k = self.transformer_score(k, self.num_attention_head, attention_size)
        v = self.transformer_score(v, self.num_attention_head, attention_size)
        # q*kt
        qk = np.matmul(q, k.swapaxes(1, 2))
        qk /= np.sqrt(attention_size)
        qk = softmax(qk)
        qkv = np.matmul(qk,v)
        qkv = qkv.swapaxes(0, 1).reshape(-1, self.hidden_size)
        attention_output = np.dot(qkv, attention_output_weight.T) + attention_output_bias
        return attention_output
    # 多头机制
    def transformer_score(self,x, num_attention_head, attention_size):
        max_len,hidden_size = x.shape
        x = x.reshape(max_len, num_attention_head, attention_size)
        x = x.swapaxes(1, 0)
        return x

    # feed_forward
    def feed_forward(self,x,intermediate_weight, intermediate_bias,output_weight, output_bias):
        feed_x = np.dot(x,intermediate_weight.T) + intermediate_bias
        feed_x = gelu(feed_x)
        feed_x = np.dot(feed_x,output_weight.T) + output_bias
        return feed_x

    def pool_output(self, x):
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        x = np.tan(x)
        return x
    def forward(self, x):
        # embedding层
        x = self.embedding_forward(x)
        x = self.all_transformer(x)
        pool_output = self.pool_output(x[0])
        return x,pool_output


# if __name__ == '__main__':
#     diyBert = DiyModel(state_dict=bert_static_dict)
#     sequence_output1, pooler_output1 = diyBert.forward(x)
#     print(pooler_output1)
#     sequence_output2, pooler_output2 = bert(torch_x)
#     print(pooler_output2)
#     print()



"""
embedding层:
token embedding 权重   vocab_size * hidden_size 
segment embedding 权重 2 * hidden_size
position embedding 权重 512 * hidden_size
layer normal 权重: 
weight : hidden_size
bias : hidden_size

#transformer权重
q权重 weight,bias   hidden_size * hidden_size hidden_size
k权重 weight,bias hidden_size * hidden_size hidden_size
v权重 weight，bias hidden_size * hidden_size hidden_size
qkv以后线性层权重 :weight,bias  hidden_size * hidden_size, hidden_size
layer normal: weight ,bias 权重 hidden_size, hidden_size

feed forward权重:
第一个线性层: first_weight first_bias   hidden_size * 3072  3072
第二个线性层: second_weight second_bias 3072 *  hidden_size hidden_size
layer normal: weight ,bias 权重 hidden_size,hidden_size

pool层权重: weight,bias  hidden_size * hidden_size ， hidden_size
"""

# 计算bert的训练参数量，
def get_bert_weight_num():
    intermediate_size = 3072
    vocab_size = 21128
    hidden_size = 768
    max_position_embeddings = 512
    type_vocab_size = 2
    num_hidden_layers = 1
    num_attention_heads = 12
    # embedding权重

    token_embedding_weight = vocab_size * hidden_size
    type_embedding = type_vocab_size * hidden_size
    position_embedding = max_position_embeddings * hidden_size

    # embedding层 归一化weight和bias
    embedding_layer_norm_weight = hidden_size
    embedding_layer_norm_bias = hidden_size

    size = token_embedding_weight + type_embedding + position_embedding + embedding_layer_norm_weight + embedding_layer_norm_bias

    # transformer
    q_weight = hidden_size * hidden_size * num_hidden_layers
    q_bias = hidden_size  * num_hidden_layers

    k_weight = hidden_size * hidden_size  * num_hidden_layers
    k_bias = hidden_size   * num_hidden_layers

    v_weight = hidden_size * hidden_size  * num_hidden_layers
    v_bias = hidden_size  * num_hidden_layers

    size = size + q_weight + q_bias + k_weight + k_bias + v_weight + v_bias

    # qkv以后得线性层
    qkv_weight = hidden_size * hidden_size  * num_hidden_layers
    qkv_bias = hidden_size  * num_hidden_layers
    # layer_normal权重
    attention_layer_weight = hidden_size  * num_hidden_layers
    attention_layer_bias = hidden_size  * num_hidden_layers

    # feed_forward
    feed_forward_first_weight = hidden_size * intermediate_size * num_hidden_layers
    feed_forward_first_bias = intermediate_size * num_hidden_layers

    feed_forward_second_weight = intermediate_size * hidden_size * num_hidden_layers
    feed_forward_second_bias = hidden_size * num_hidden_layers

    feed_forward_layer_weight = hidden_size  * num_hidden_layers
    feed_forward_layer_bias = hidden_size  * num_hidden_layers

    # pool
    pool_weight = hidden_size * hidden_size * num_hidden_layers
    pool_bias = hidden_size

    size = size + qkv_weight + qkv_bias + attention_layer_weight + attention_layer_bias + \
           feed_forward_first_weight + feed_forward_first_bias + feed_forward_second_weight + feed_forward_second_bias \
           + feed_forward_layer_weight + feed_forward_layer_bias +  pool_weight +  pool_bias
    return size


if __name__ == '__main__':
    size = get_bert_weight_num()
    print(size)
