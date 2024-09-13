# coding: utf-8
"""
计算bert可训练参数
"""
from transformers import BertModel

model = BertModel.from_pretrained(r"D:\plugin\bert-base-chinese\bert-base-chinese", return_dict=False)
sentence_count = 2  # 输入句子个数
vocab_size = 21128  # 词表大小
max_sequence_length = 512  # 最大序列长度
embedding_size = 768  # 词向量维度
hidden_size = 3072  # 隐藏层维度

# embedding参数
# 词表embedding参数：vocab_size * embedding_size
# 位置embedding参数：max_sequence_length * embedding_size
# 句子embedding参数：sentence_count * embedding_size
# layer_norm层参数：embedding_size * 2
embedding_parameters = vocab_size * embedding_size + max_sequence_length * embedding_size + sentence_count * embedding_size + embedding_size + embedding_size
print("embedding_parameters:%d" % embedding_parameters)

# encoder参数
# self_attention参数：
# 1. Q矩阵：embedding_size * embedding_size + embedding_size
# 2. K矩阵：embedding_size * embedding_size + embedding_size
# 3. V矩阵：embedding_size * embedding_size + embedding_size
self_attention_parameters = (embedding_size * embedding_size + embedding_size) * 3
print("self_attention_parameters:%d" % self_attention_parameters)

# self_attention_output参数：
# 1. 第一层线性层：embedding_size * embedding_size
# 2. 第二层线性层：embedding_size
# 3. 第三层线性层：embedding_size
# 4. 残差映射偏置：embedding_size + embedding_size
self_attention_output_parameters = embedding_size * embedding_size + embedding_size + embedding_size + embedding_size
print("self_attention_output_parameters:%d" % self_attention_output_parameters)

# feed_forward参数：
# 1. 第一个线性层：hidden_size * embedding_size + hidden_size
# 2. 第二个线性层：embedding_size * hidden_size + embedding_size
# 3. 残差映射矩阵：embedding_size + embedding_size
feed_forward_parameters = (hidden_size * embedding_size + hidden_size
                           + embedding_size * hidden_size + embedding_size
                           + embedding_size + embedding_size)
print("feed_forward_parameters:%d" % feed_forward_parameters)

# pool_output参数：
# 1. 线性映射矩阵：embedding_size * embedding_size
# 2. 线性映射偏置：embedding_size
pool_fc_parameters = embedding_size * embedding_size + embedding_size
print("pool_fc_parameters:%d" % pool_fc_parameters)

# 模型总参数
model_parameters = (embedding_parameters + self_attention_parameters
                    + self_attention_output_parameters + feed_forward_parameters + pool_fc_parameters)


print("Bert模型实际参数个数为%d" % sum(p.numel() for p in model.parameters()))
print("DIY计算可训练参数个数为%d" % model_parameters)
