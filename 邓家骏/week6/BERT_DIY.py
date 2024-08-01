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
lnw = state_dict['embeddings.LayerNorm.weight']
qw = state_dict['encoder.layer.0.attention.self.query.weight']

bert.eval()
x = np.array([2213,13341,152,2221])
torch_x = torch.LongTensor([x])

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
        ps:对行减均值，除方差。（作用是方便后续计算，避免出现数值极端导致报错异常（梯度爆炸梯度消失？））
        ps2:记得有对列求均值的情况？pooling?用于描述整体+简化计算
        ps3:norm的参数量取决于？ 公式是x*w + b，参数量取决于hidden。
            注意：norm的最后一步不是线性层的计算。称为缩放与平移(gamma,beta)
        (
            (x_emb + x_attention) - mean((x_emb + x_attention),dim = 1,keepdims = True)
        ) / np.std(x,dim = 1, keepdims = True)

        feed forward
        linear -> relu -> linear
        FFN(x) = max(0,x * W1 + B1) * W2 + B2

        再过一层Add & norm
        这里Add的是FFN之前的x
        


    
'''

# 写外面是因为他不属于这个bert模型的实现，虽然看着难受。
# 公式：[e^x1,e^x2,e^x3...]/(e^x1+e^x2+e^3...)
def softmax(x):
    # 这里axis = -1，就是自动对最后一维求和
    return np.exp(x)/np.sum(np.exp(x),axis=-1,keepdims=True)

# 同上
# 公式：x * cdf(x) 算了吧hh
# 有近似公式：0.5x(1+tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * np.power(x,3))))
class DiyBert:
    def __init__(self,state_dict):
        # 由预训练config.json决定
        self.num_layers = 1
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.weight_load(state_dict)

    def weight_load(self,state_dict):
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
        self.pooler_w = state_dict['pooler.dense.weight'].numpy()
        self.pooler_b = state_dict['pooler.dense.bias'].numpy()

    def embedding_forward(self,x):
        we,pe,se = self.get_emb(x)
        emb = we + pe + se
        
        # 过一层layer norm
        x = self.layer_norm(emb,self.emb_layerNorm_w,self.emb_layerNorm_b)
        return x
    
    def all_transformer_layer_forward(self,x):
        for i in range(self.num_layers):
            x = self.transformer_forward(x,i)
        return x
    
    def transformer_forward(self,x,layer_idx):
        # 先把参数取出来
        q_w,q_b,k_w,k_b,v_w,v_b,attention_output_w,attention_output_b, \
        attention_layer_norm_w,attention_layer_norm_b,ff_intermediate_w,ff_intermediate_b,ff_output_w,ff_output_b, \
        ff_layer_norm_w,ff_layer_norm_b = self.transformer_weight[layer_idx]

        x_attention = self.self_attention(x,q_w,q_b,k_w,k_b,v_w,v_b,
                                          attention_output_w,attention_output_b,self.num_attention_heads,self.hidden_size)
        x = self.layer_norm(x + x_attention,attention_layer_norm_w,attention_layer_norm_b)

        x_FFN = self.layer_FFN(x,ff_intermediate_w,ff_intermediate_b,ff_output_w,ff_output_b)
        x = self.layer_norm(x + x_FFN,ff_layer_norm_w,ff_layer_norm_b)

        return x
    

    def get_emb(self,x):
        l = len(x)
        we = np.array([self.word_embeddings[i] for i in x])
        pe = np.array([self.position_embeddings[i] for i in range(l)])
        # 先默认单输入，即seg为0
        se = np.array([self.type_embeddings[int(i)] for i in np.zeros(l)])
        return we,pe,se
    
    # 减均值除方差
    def layer_norm(self,x,w,b):
        x = (x - np.mean(x,axis=1,keepdims=True)) / np.std(x,axis=1,keepdims=True)
        x = x * w + b
        return x
    
    def self_attention(self,x,q_w,q_b,k_w,k_b,v_w,v_b,
                       attention_output_weight,
                       attention_output_bias,
                       num_attention_heads,
                       hidden_size):
        # 算qkv
        q = np.dot(x,q_w.T) + q_b
        k = np.dot(x,k_w.T) + k_b
        v = np.dot(x,v_w.T) + v_b

        # 切片
        q = self.split_multi_head(q,num_attention_heads)
        k = self.split_multi_head(k,num_attention_heads)
        v = self.split_multi_head(v,num_attention_heads)

        # qvk切完并行算
        # np.matmul。矩阵运算，我理解高维数组做矩阵运算，只有最后两维被视为矩阵，前面的会视为批次。
        qk = np.matmul(q,k.swapaxes(1,2))
        qk /= np.sqrt(self.hidden_size/self.num_attention_heads)
        qk = softmax(qk)
        
        qkv = np.matmul(qk,v)
        qkv = qkv.swapaxes(1,0).reshape(-1,hidden_size)

        # 过一层线性层
        x = np.dot(qkv, attention_output_weight.T) + attention_output_bias

        return x
    
    def split_multi_head(self,x,num_attention_heads):
        len,hidden = x.shape
        x = x.reshape(len,num_attention_heads,-1)
        x = x.swapaxes(1,0)
        return x
    
    # ff_intermediate_w = h * 4h 
    def layer_FFN(self,x,ff_intermediate_w,ff_intermediate_b,ff_output_w,ff_output_b):
        x = np.dot(x,ff_intermediate_w.T) + ff_intermediate_b
        x = gelu(x)
        x = np.dot(x,ff_output_w.T) + ff_output_b
        return x

    def pooler_output_layer(self,x):
        x = np.dot(x,self.pooler_w.T) + self.pooler_b
        x = np.tanh(x)
        return x
    
    def forward(self,x):
        x = self.embedding_forward(x)
        sequence_output = self.all_transformer_layer_forward(x)
        # 因为是单输入。批量样本是：sequence_output[:0:],提取句首符，经过多层转换后可以代表整个句子
        pooler_output = self.pooler_output_layer(sequence_output[0])
        return sequence_output,pooler_output

# 手动实现
my_bert = DiyBert(state_dict)
my_sequence_output,my_pooler_output = my_bert.forward(x)

# torch
sequence_output,pooler_output = bert(torch_x)
sequence_output = sequence_output[0]
pooler_output = pooler_output[0]
# print(my_sequence_output,my_pooler_output)
# print(sequence_output,pooler_output)

print(my_sequence_output)
print(sequence_output)