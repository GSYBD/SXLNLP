n = 2  # 输入最大句子个数
vocab = 21128  # 词表数目
max_sequence_length = 512  # 最大句子长度
embedding_size = 768  # embedding维度
hide_size = 3072  # 隐藏层维数

# Embedding
embedding_parameters = max_sequence_length * embedding_size + n * embedding_size + max_sequence_length * embedding_size
# lay_norm:embedding_size + embedding_sizes
embedding_parameters += embedding_size + embedding_size

# self_attention:*3分别是Q,K,V,3个线性层
self_attention_parameters = (embedding_size * embedding_size + embedding_size) * 3
# 一个线性层+一个lay_norm
self_attention_out_parameters = embedding_size * embedding_size + embedding_size + embedding_size + embedding_size

# Feed-Forward:前馈神经网络，两个线性层，一个lay_norm
feed_forward_parameters = embedding_size * hide_size + hide_size + embedding_size * hide_size + embedding_size + embedding_size + embedding_size

# pool_fc层参数,使用第一个向量
pool_fc_parameters = embedding_size * embedding_size + embedding_size

# 总参数数目
total_parameters = embedding_parameters + self_attention_parameters + self_attention_out_parameters + feed_forward_parameters + pool_fc_parameters

print("Total parameters:", total_parameters)
