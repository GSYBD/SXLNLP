

# embedding
words_embeddings_weight = 21128 * 768
position_embeddings_weight = 512 * 768  
token_type_embeddings_weight = 2 * 768
layer_norm_weight = 768
layer_norm_bias = 768

# Q,K,V   输入x = sentence_length x hidden_size(768)
query_weight = 768 * 768
query_bias = 768
key_weight = 768 * 768
key_bias = 768
value_weight = 768 * 768
value_bias = 768

# 𝑜𝑢𝑡𝑝𝑢𝑡=𝐿𝑖𝑛𝑒a𝑟(𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛(𝑄,𝐾,𝑉))  经过线性层
attention_output_dense_weight = 768 * 768
attention_output_dense_bias = 768

# LayerNorm(Xembedding+ Xattention)
attention_layer_norm_weight = 768
attention_layer_norm_bias = 768

# 𝑜𝑢𝑡𝑝𝑢𝑡=𝐿𝑖𝑛𝑒a𝑟(𝑔𝑒𝑙𝑢(𝐿𝑖𝑛𝑒a𝑟(𝑥)))
intermediate_dense_weight = 768 * (768 * 4)
intermediate_dense_bias = 768 * 4
output_dense_weight = (768 * 4) * 768
output_dense_bias = 768

# LayerNorm(X forward+ X attention)
output_layer_norm_weight = 768
output_layer_norm_bias = 768

# BERT模型在经过多个编码层之后，会经过一个特殊的Pooler层，用于从序列中提取一个固定长度的特征向量。这个特征向量可以用于分类任务或其他任务。
# 在BERT模型中，Pooler层通常是一个多层感知器（MLP），它接受编码层的输出作为输入，并输出一个固定长度的特征向量。
pooler_dense_weight = 768 * 768
pooler_dense_bias = 768


