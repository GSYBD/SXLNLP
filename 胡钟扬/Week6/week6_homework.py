import torch
import math
import numpy as np
from transformers import BertModel






def main():
    bert = BertModel.from_pretrained(r"D:\pre-trained-models\bert-base-chinese", return_dict=False)

    state_dict = bert.state_dict()
    
    bert.eval()
    
    x = np.array([2450, 15486, 102, 2110])
    torch_x = torch.LongTensor([x])

    print(torch_x.shape)
    
    sequence_output, pooler_output = bert(torch_x)

    print(sequence_output.shape, pooler_output.shape)

    print(bert.state_dict().keys())
    

    




def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))



def gelu(x):
    '''
        激活函数
    
    '''
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))





class DiyBert:
    
    def __init__(self,state_dict):
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 1
        
        # 计算bert中的总参数量
        self.total_parameter_num = 0
        self.load_weights(state_dict)
    
    def load_weights(self,state_dict):
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()
        self.positional_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
    
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
        
        print("embedding layer norm weight shape = ",self.embeddings_layer_norm_weight.shape)
        
        embedding_params = self.word_embeddings.shape[0]*self.word_embeddings.shape[1]\
            + self.positional_embeddings.shape[0]*self.positional_embeddings.shape[1]\
            + self.token_type_embeddings.shape[0]*self.token_type_embeddings.shape[1]\
            + self.embeddings_layer_norm_weight.shape[0]\
            + self.embeddings_layer_norm_bias.shape[0]
        
        # 统计 embedding层参数量
        self.total_parameter_num+= embedding_params
        self.transformer_weights = []
        
        
        for i in range(self.num_layers):
            
            
            # Q K V
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy()
            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy()
            k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy()
            k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy()
            v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy()
            v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy()
            

            # 计算QKV的总参数量
            params_q_w = q_w.shape[0]*q_w.shape[1]    
            params_q_b = q_b.shape[0]
            
            params_k_w = k_w.shape[0]*k_w.shape[1]
            params_k_b = k_b.shape[0]
            
            params_v_w = v_w.shape[0]*v_w.shape[1]
            params_v_b = v_b.shape[0]


            params_qkv = params_q_w + params_q_b + params_k_w + params_k_b + params_v_w + params_v_b
    
            
            # z矩阵 dense指的是注意力机制输出部分的一个线性变换层
            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy()
            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy()
            
            attention_output_params = attention_output_weight.shape[0]*attention_output_weight.shape[1]\
                + attention_output_bias.shape[0]
                
             # LayerNorm(Xembedding + Xattention)  减均值，除方差，乘w+b
            attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy()
            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy()

            # 统计 attention_layer_norm的参数量
            attention_layer_norm_params = attention_layer_norm_w.shape[0]\
                + attention_layer_norm_b.shape[0]
                
            # 线性层 ffn1 的权重
            intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy()
            intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy()
            # 统计 intermediate的参数量
            intermediate_params = intermediate_weight.shape[0]*intermediate_weight.shape[1]\
                + intermediate_bias.shape[0]

            # 线性层 ffn2 的权重
            output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy()
            output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy()
            # 统计 output的参数量
            output_params = output_weight.shape[0]*output_weight.shape[1]\
                + output_bias.shape[0]
                
            # LayerNorm(output+z) 中的w和b权重
            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy()
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()
            # 统计 ff_layer_norm的参数量
            ff_layer_norm_params = ff_layer_norm_w.shape[0]\
                + ff_layer_norm_b.shape[0]
                
            # 计算transformer_layer的参数量
            self.total_parameter_num += params_qkv + attention_output_params + attention_layer_norm_params\
                + intermediate_params + output_params + ff_layer_norm_params
            
            
            self.transformer_weights.append([q_w,q_b,k_w,k_b,v_w,v_b,attention_output_weight,attention_output_bias,
                                            intermediate_weight,intermediate_bias,output_weight,output_bias,
                                            attention_layer_norm_w,attention_layer_norm_b,ff_layer_norm_w,ff_layer_norm_b])

        # pooler层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()
        pooler_params = self.pooler_dense_weight.shape[0]*self.pooler_dense_weight.shape[1]\
            + self.pooler_dense_bias.shape[0]
        
        # 统计pooler的参数量
        self.total_parameter_num+= pooler_params  

    def embedding_forward(self, x):
        we= self.get_embedding(self.word_embeddings, x)
        
        pe = self.get_embedding(self.positional_embeddings, list(range(len(x))))
        
        te = self.get_embedding(self.token_type_embeddings, [0]*len(x))
    
        embedding = we + pe + te
        
        embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)
        return embedding
    
    
    def get_embedding(self, embedding_matrix, x):
        
       return [embedding_matrix[index] for index in x]
        
    
    def get_parameter_number(self, x):
       pass
    
    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x, i)
        return x
    
    
    def single_transformer_layer_forward(self, x, layer_index):
        weights = self.transformer_weights[layer_index]
        
        q_w, q_b,\
        k_w, k_b,\
        v_w, v_b,\
        attention_output_weight, attention_output_bias,\
        intermediate_weight, intermediate_bias,\
        output_weight, output_bias,\
        attention_layer_norm_w, attention_layer_norm_b,\
        ff_layer_norm_w, ff_layer_norm_b = weights
        
        
        attention_output = self.self_attention(x,
                                        q_w,q_b,
                                        k_w, k_b,
                                        v_w, v_b,
                                        attention_output_weight, attention_output_bias,
                                        self.num_attention_heads,
                                        self.hidden_size)
        # 过一层 layer_norm, 使用了残差机制
        x  = self.layer_norm(x+attention_output, attention_layer_norm_w, attention_layer_norm_b)
        
        # 过feedforward
        feed_forward_x = self.feedforward(x, intermediate_weight, intermediate_bias, output_weight, output_bias)
        
        
        # 再过一层 layer_norm, 使用了残差机制
        x = self.layer_norm(feed_forward_x + x, ff_layer_norm_w, ff_layer_norm_b)
        return x
    
    def self_attention(self,
                        x,
                        q_w,
                        q_b,
                        k_w,
                        k_b,
                        v_w,
                        v_b,
                        attention_output_weight,  # hidden_size, hidden_size
                        attention_output_bias,  # hidden_size
                        num_attention_heads,
                        hidden_size):
        
        # x: max_len x hidden_size
        print("x.shape = ",x.shape)
        print("q_w.shape = ",q_w.shape)
        print("q_b.shape = ",q_b.shape)
        
        q = np.dot(x,q_w.T) + q_b # [max_len x hidden_size]
        
        k = np.dot(x, k_w.T) + k_b
        
        v = np.dot(x, v_w.T) + v_b
        
        attention_head_size = int(hidden_size/num_attention_heads)
        
        # 根据多头对qkv进行分块 [num_attention_heads x max_len x attention_head_size]
        q = self.multi_head_attention(q,num_attention_heads,attention_head_size)
        
        k = self.multi_head_attention(k, num_attention_heads, attention_head_size)
        
        v = self.multi_head_attention(v, num_attention_heads, attention_head_size)

        # 在每个head上面进行attention计算  [num_attention_heads x max_len x max_len]
        qk = np.matmul(q, k.swapaxes(1,2)) # head维不动，剩余两维进行矩阵乘法
        qk = qk/np.sqrt(attention_head_size)
        
        qk = softmax(qk) 
        
        # 对于3维矩阵乘法，[d1, d2, d3] x [d1, d3, d4] = [d1, d2, d4]
        # 其本质，是逐批次计算二维矩阵，d1是批次，其本质还是[d2, d3] x [d3, d4]
        qkv = np.matmul(qk, v) # [num_attention_heads x max_len x attention_head_size]
        
        
        # 把所有的attention 头合并到一起
        qkv = np.swapaxes(qkv,0,1) #[max_len x num_attention_heads x attention_head_size]
        print("qkv.shape = ",qkv.shape)
        qkv = qkv.reshape(-1, hidden_size)
        
        
        attention = np.matmul(qkv, attention_output_weight)+attention_output_bias
        
        return attention
        
        
    
    
    def multi_head_attention(self, x, num_attention_heads, attention_head_size):
        max_len, hidden_size = x.shape
        
        x = x.reshape(max_len, num_attention_heads, attention_head_size)
        
        x = x.swapaxes(0,1)
        
        return x
    
    
    
    
    
                   
    def feedforward(self,
                     x,
                     intermediate_weight,  # hidden_size, intermediate_size
                     intermediate_bias,  # intermediate_size
                     output_weight,  # intermediate_size, hidden_size
                     output_bias,  # hidden_size
                     ):
        
        # max_len x intermediate_size
        x = np.dot(x, intermediate_weight.T)+intermediate_bias
        x = gelu(x)
        
        # max_len x hidden_size
        x = np.dot(x, output_weight.T)+output_bias
        
        return x
    
    
    
    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True))/np.std(x, axis=1, keepdims=True)
        
        x =x*w+b
        return x
    
    
    def pooler_output_layer(self, x):
        '''
        
        x: [hidden_size]

        '''

        x = np.dot(x, self.pooler_dense_weight.T)+self.pooler_dense_bias

        x =np.tanh(x)
        print("pooler_output_x.shape = ", x.shape)
        
        
        return x
    
    
    def forward(self, x):
        # x.shape = [max_len]
        x =self.embedding_forward(x)
        
        sequence_output = self.all_transformer_layer_forward(x)
        print("sequence_output.shape = ", sequence_output.shape)
        pooler_output = self.pooler_output_layer(sequence_output[0])
        
        return sequence_output, pooler_output
    



if __name__ == '__main__':
    # main()
    
    bert = BertModel.from_pretrained(r"D:\pre-trained-models\bert-base-chinese", return_dict=False)
    state_dict = bert.state_dict()

    db = DiyBert(state_dict)
    
    x = np.array([2450, 15486, 102, 2110])
    print("x.shape = ",x.shape)
    diy_sequence_output, diy_pooler_output = db.forward(x)
    
    print("=========================diy_sequence_output=======================")
    print(diy_sequence_output)
    print(diy_sequence_output.shape)


    print("=========================total paramter number ====================")
    print(db.total_parameter_num)
    
    
