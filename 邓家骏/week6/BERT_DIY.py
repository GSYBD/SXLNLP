import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertModel

"""
目标：
实现bert的这个算法。只算一层
直接拿已经练好的参数（模型），然后做预测，只预测输出第一层，看结果是否一致

要做的：
1.实现加载bert并只预测第一层

2.按理解拿已有的参数实现计算过程

3.做结果对比

看着不多hh
"""

model_path = r'D:\code\github\bert-base-chinese'
bert = BertModel.from_pretrained(model_path,return_dict = False)

state_dict = bert.state_dict()
bert.eval()
x = np.array([2213,13341,152,2221])
a,b= bert(torch.LongTensor([x]))
print(state_dict)

# 第一步，过embeding
# 之前对embedding有误解，以为embedding分为固定向量和训练参数We; 其实训练参数We本身就是embedding，词向量。具体训练过程不清楚（知道是老一套反向传播，但还是感觉模糊），后补。

'''
    这个类该怎么写？结构？
    init
        初始化值,读W,层数等等

    定义每一层的计算方式

    通过forward(计算，向前传播)连起来
    forward可以拆分成embedding_forward,transformer_forward

    embedding_forward:
    x_emb = token_emb + position + seg

    transformer_forward:
        q = (x * Wq) + Bq
        k = (x * Wk) + Bk
        v = (x * Wv) + Bv

        x_attention = Attention(Q,K,V) = (z_split_concat * w_attention) + b_attention
        * z_split.shape 应该为 num_attention_heads,max_len,hidden_size/num_attention_heads
        * z_split_concat 为 max_len,hidden_size
        z_split = softmax(
            (q_split * k_split_T)/sqrt( hidden_size/num_attention_heads)
        ) * v_split

        Add & norm,与计算前的x求和，在归一化（减均值求方差）:
        ps:对行求均值，减方差。（作用是方便后续计算，避免出现数值极端导致报错异常（梯度爆炸梯度消失？））
        ps2:记得有对列求均值的情况？pooling?用于描述整体+简化计算
        (
            (x_emb + x_attention) - mean((x_emb + x_attention),dim = 1,keepdims = True)
        ) / np.std(x,dim = 1, keepdims = True)

        feed forward
        linear -> relu -> linear
        FFN(x) = max(0,x * W1 + B1) * W2 + B2

        再过一层Add & norm
        这里Add的是FFN之前的x
        


    
'''
class DiyBert:
    def __init__(self,state_dict):
        # 由预训练config.json决定
        self.num_layers = 1
        self.num_attention_heads = 12
        self.hidden_size = 768
        # 要读哪几个参数？
        # embedding -> 词向量 + position + seg
        self.word_embeddings = state_dict['embeddings.word_embeddings.weight'].numpy()
        self.position_embeddings = state_dict['embeddings.position_embeddings.weight'].numpy()
        self.type_embeddings = state_dict['embeddings.token_type_embeddings.weight'].numpy()
        self.emb_layerNorm_w = state_dict['embeddings.LayerNorm.weight'].numpy()
        self.emb_layerNorm_b = state_dict['embeddings.LayerNorm.bias'].numpy()
        self.transformer_weight = []
        for i in range(self.num_layers):
            q_w = state_dict['encoder.layer.%d.attention.self.query.weight' % i].numpy()
            q_b = state_dict['encoder.layer.%d.attention.self.query.bias' % i].numpy()
            k_w = state_dict['encoder.layer.%d.attention.self.key.weight' % i].numpy()
            k_b = state_dict['encoder.layer.%d.attention.self.key.bias' % i].numpy()
            v_w = state_dict['encoder.layer.%d.attention.self.value.weight' % i].numpy()
            v_b = state_dict['encoder.layer.%d.attention.self.value.bias' % i].numpy()
            attention_output_w = state_dict['encoder.layer.%d.attention.output.dense.weight' % i].numpy()
            attention_output_b = state_dict['encoder.layer.%d.attention.output.dense.bias' % i].numpy()
            attention_layer_norm_w = state_dict['encoder.layer.%d.attention.output.LayerNorm.weight' % i].numpy()
            attention_layer_norm_b = state_dict['encoder.layer.%d.attention.output.LayerNorm.bias' % i].numpy()
            ff_intermediate_w = state_dict['encoder.layer.%d.intermediate.dense.weight' % i].numpy()
            ff_intermediate_b = state_dict['encoder.layer.%d.intermediate.dense.bias' % i].numpy()
            ff_output_w = state_dict['encoder.layer.%d.output.dense.weight' % i].numpy()
            ff_output_b = state_dict['encoder.layer.%d.output.dense.bias' % i].numpy()
            ff_layer_norm_w = state_dict['encoder.layer.%d.output.LayerNorm.weight' % i].numpy()
            ff_layer_norm_b = state_dict['encoder.layer.%d.output.LayerNorm.bias' % i].numpy()
            self.transformer_weight.append([q_w,q_b,k_w,k_b,v_w,v_b,attention_output_w,attention_output_b,
                                       attention_layer_norm_w,attention_layer_norm_b,ff_intermediate_w,ff_intermediate_b,
                                       ff_output_w,ff_output_b,ff_layer_norm_w,ff_layer_norm_b])
            
        # 这个是？
        self.pooler_w = state_dict['pooler.dense.weight']
        self.pooler_b = state_dict['pooler.dense.bias']

    def embedding_forward(self,x):
        we,pe,se = self.get_emb(x)
        emb = we + pe + se
        
        # 过一层layer norm
        x = self.layer_norm(emb,self.emb_layerNorm_w,self.emb_layerNorm_b)
        return x
    
    def transformer_forward(self,x,layer_idx):
        # 先把参数取出来
        q_w,q_b,k_w,k_b,v_w,v_b,attention_output_w,attention_output_b, \
        attention_layer_norm_w,attention_layer_norm_b,ff_intermediate_w,ff_intermediate_b,ff_output_w,ff_output_b, \
        ff_layer_norm_w,ff_layer_norm_b = self.transformer_weight[layer_idx]

        self.self_attention()



        return x
    

    def get_emb(self,x):
        l = len(x)
        we = np.array([self.word_embeddings[i] for i in x])
        pe = np.array([self.position_embeddings[i] for i in range(l)])
        # 先默认单输入，即seg为0
        se = np.array([self.type_embeddings[i] for i in np.zeros(l)])
        return we,pe,se
    
    # 减均值除方差
    def layer_norm(self,x,w,b):
        x = (x - np.mean(x,axis=1,keepdims=True)) / np.std(x,axis=1,keepdims=True)
        x = x * w + b
        return x
    
    def self_attention(self,q_w,q_b,k_w,k_b,v_w,v_b):
        # 算qkv
        # ps，这里矩阵乘法有问题，示例代码，归一化层用*,但是self_attention用np.dot()
        q = x * q_w + q_b
        k = x * k_w + k_b
        v = x * v_w + v_b

        # qvk切完并行算
        # 

        x_attention = 0

        return x_attention
