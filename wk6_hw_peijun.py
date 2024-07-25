type_vocab_size = 2 #句子类别数
num_hidden_layers = 12 #层数
hidden_size = 768
vocab_size = 21128 #词表大小
max_len = 200
max_len = min(max_len,512) #输入文本的最长长度

# embedding层
token_embedding = vocab_size * hidden_size
segment_embedding = type_vocab_size * hidden_size
position_embedding = max_len * hidden_size
embeddings_layer_norm_weight = hidden_size * hidden_size
embeddings_layer_norm_bias = max_len * hidden_size
embeddings_layer_norm = embeddings_layer_norm_weight + embeddings_layer_norm_bias
embeddings_layer_params = token_embedding + segment_embedding + position_embedding + embeddings_layer_norm
#公式推导
embeddings_layer_params_func = vocab_size * hidden_size + type_vocab_size * hidden_size + max_len * hidden_size + hidden_size * hidden_size + max_len * hidden_size
embeddings_layer_params_func = (vocab_size + type_vocab_size + 2*max_len + hidden_size) * hidden_size

#每层 self_attention
q_w = hidden_size * hidden_size
q_b = max_len * hidden_size
k_w = hidden_size * hidden_size
k_b = max_len * hidden_size
v_w = hidden_size * hidden_size
v_b = max_len * hidden_size
#attention层的线性层
attention_linear_weight = hidden_size * hidden_size
attention_linear_bias = max_len * hidden_size
#attention层的归一化
attention_layer_norm_w = hidden_size * hidden_size
attention_layer_norm_b = max_len * hidden_size
#attention层参数个数汇总
attention_layer_params = q_w + q_b + k_w + k_b + v_w + v_b + attention_linear_weight + attention_linear_bias + attention_layer_norm_w +attention_layer_norm_b
#公式推导
attention_layer_params_func = 3*hidden_size*hidden_size + 3*max_len*hidden_size \
                         + 2*hidden_size*hidden_size + 2*max_len*hidden_size
attention_layer_params_func = 5*hidden_size*hidden_size + 5*max_len*hidden_size


#每层feed_forward
#里层线性
intermediate_weight = hidden_size * (4*hidden_size)
intermediate_bias = max_len * (4*hidden_size)
#外层线性
output_weight = hidden_size * (4*hidden_size)
output_bias = max_len * (4*hidden_size)
#feed_forward层的归一化
ff_layer_norm_w = hidden_size * hidden_size
ff_layer_norm_b = max_len * hidden_size
# feed_forword层参数个数汇总
ff_layer_params = intermediate_weight + intermediate_bias + output_weight + output_bias + ff_layer_norm_w + ff_layer_norm_b
ff_layer_params_func = 2 * (hidden_size+max_len) * (4*hidden_size) + (hidden_size+max_len)*hidden_size
ff_layer_params_func = 9 * (hidden_size+max_len) * hidden_size

#计算总共的参数个数
#embedding层
embeddings_layer_params = embeddings_layer_params * 1
#self_attention层及feed_forward层
attention_and_ff_param = (attention_layer_params + ff_layer_params) * num_hidden_layers
#总参数个数
total_params_num = embeddings_layer_params + attention_and_ff_param
print('总参数个数为：',total_params_num)

total_params_num_func = (vocab_size + type_vocab_size + 2*max_len + hidden_size) * hidden_size \
                        + (5*hidden_size*hidden_size + 5*max_len*hidden_size\
                        + 9*(hidden_size+max_len) * hidden_size) * num_hidden_layers

total_params_num_func = vocab_size*hidden_size + type_vocab_size*hidden_size + 2*max_len*hidden_size + hidden_size^2 \
                      + 14*num_hidden_layers*hidden_size^2 + 14*num_hidden_layers*hidden_size
total_params_num_func = (vocab_size+type_vocab_size+2*max_len+14*num_hidden_layers)*hidden_size \
                         +(14*num_hidden_layers+1)*hidden_size^2
print('计算bert可训练参数个数公式',"(vocab_size+type_vocab_size+2*max_len+14*num_hidden_layers)*hidden_size \
                         +(14*num_hidden_layers+1)*hidden_size^2")




