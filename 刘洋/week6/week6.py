#week6 计算transformer模型参数量

# 训练参数
sentence_len = 512
batches = 1000
vocab_len = 21128
embedding_dim = 768
hidden_dim = 768
layer_num = 12
type_vocab_size=2
pooler_num_fc_layers=3

# 过三层embedding层
# 第一层embedding
embedding_case1 = vocab_len * embedding_dim
# 第二层embedding
embedding_case2=type_vocab_size * embedding_dim
# 第三层embedding
embedding_case3=sentence_len * embedding_dim
# 把三层embedding训练数相加
embedding_sum = embedding_case1 + embedding_case2+embedding_case3


# 然后进入transformer结构，使用multi_head self_attention
norm_args = 2 * embedding_dim
# 然后进入transformer结构，使用multi_head self_attention
transformer_sum = 0
# 把embedding训练数相加的结果分别送到Q,K,V中
Q_b_shape = embedding_sum
K_b_shape = embedding_sum
dim_v = embedding_sum
# 得到self_attention训练数
attention_sum = pooler_num_fc_layers * embedding_dim * hidden_dim + pooler_num_fc_layers * hidden_dim


# 因为需要叠加多层，所以hidden_dim==embedding_dim 便于多次注入
# 多头注意力的分配权重dense层，linear结构
attention_output_dense_sum = hidden_dim ** 2 + hidden_dim
# 残差网络+归一化
# args += norm_args
transformer_sum += (attention_sum + norm_args + attention_output_dense_sum)
# FFN层+残差&归一化(Add&Norm)
intermediate_dim = 4 * hidden_dim
FFN_args = (hidden_dim * intermediate_dim + intermediate_dim
            + intermediate_dim * embedding_dim + embedding_dim)
transformer_sum += (FFN_args + norm_args)
sum = embedding_sum + layer_num * transformer_sum
# 最后的pooler层
pooler_dense_args = hidden_dim * hidden_dim + hidden_dim
sum = embedding_sum + layer_num * transformer_sum + pooler_dense_args



print(sum)