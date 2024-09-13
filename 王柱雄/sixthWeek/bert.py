# 词表大小为5200，最长序列长度为500，隐藏层维度768

# embedding
embedding_num = 5200 * 768 + 500 * 768 + 2 * 768

# self-Attention
self_attention_num = 4 * 768 * 768

# feed-forward
feed_forward_num = 2 * 768 * 768 * 4

# 共12层
num = 12 * (self_attention_num + feed_forward_num)

# 每层归一化 缩放因子scale：维度为 768, 偏置b：维度为 768
normalization_num = 2 * 768
# 共12层
normalization_num = normalization_num * 12

# 可训练参数总数
trainable_parameter_num = embedding_num + num + normalization_num

print(trainable_parameter_num)

