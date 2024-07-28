"num_hidden_layers" = 1 #按照1个transformer层进行分析

#embedding层：token_embeddings + segment_embeddings + position_embeddings

token_embeddings = vocab_size * hidden_size  #可训练参数形状
segment_embeddings = 2 * hidden_size
position_embeddings = 512 * hidden_size

trainable_parameters_num1 = vocab_size * hidden_size + 2 * hidden_size + 512  * hidden_size
#embedding = token_embeddings + segment_embeddings + position_embeddings后的输出，输出的维度为sen_len*hidden_size，然后过归一化层

#self-attention, x为embedding的输出，维度为sen_len*hidden_size，分别过三个不同的线性层
# 线性层的W形状为hidden_size*hidden_size，b的形状为sen_len*hidden_size
x * Wq = Q
x * Wk = K
x * Wv = V
# Attention(Q,K,V) = softmax(Q * KT / dk ** -0.5) * V 形状为sen_len*hidden_size，再进入一个线性层 output = Liner(Attention(Q,K,V))
# 线性层的W形状为hidden_size*hidden_size，b的形状为sen_len*hidden_size
trainable_parameters_num2 = 4 * (hidden_size * hidden_size + sen_len * hidden_size)

#和embedding层相加后再做归一化
#feed-forward
output = Liner(gelu(Liner(x)))
# 内部线性层的W形状为hidden_size * 4hidden_size，b的形状为sen_len * 4hidden_size
# 外部线性层的W形状为4hidden_size * hidden_size，b的形状为sen_len * hidden_size
# 然后与LayerNorm(Xembedding + Xattention)做残差过归一化，第一层transformer结束
trainable_parameters_num3 = hidden_size * 4hidden_size + sen_len * 4hidden_size + 4hidden_size * hidden_size + sen_len * hidden_size

# 一个transformer层所需要的可训练参数数量为：
trainable_parameters_num = trainable_parameters_num1 + trainable_parameters_num2 + trainable_parameters_num3
                         = vocab_size * hidden_size + 2 * hidden_size + 512 * hidden_size + 4 * (hidden_size * hidden_size + sen_len * hidden_size) + hidden_size * 4hidden_size + sen_len * 4hidden_size + 4hidden_size * hidden_size + sen_len * hidden_size
                         = vocab_size * hidden_size + 2 * hidden_size + 512 * hidden_size + 5 * (sen_len * hidden_size) + 4 * (hidden_size * hidden_size) + 2 * (hidden_size * 4hidden_size) + sen_len * 4hidden_size
