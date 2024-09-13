from transformers import BertModel

# 初始化BERT模型
model = BertModel.from_pretrained(r"/Users/lishuyan/PycharmProjects/lsy823/SXLNLP/李姝妍/week6/bert-base-chinese", return_dict=False)
state_dict = model.state_dict()

config = model.config
hidden_size = config.hidden_size  # 嵌入维度
vocab_size = config.vocab_size  # 词汇表大小

num_layers = config.num_hidden_layers
num_heads = 12  # 每个Transformer层的注意力头数
ff_size = 3072  # 前馈网络的中间层大小

# Embedding层参数量
embedding_params = vocab_size * hidden_size

# Transformer层参数量
# 每个自注意力层（Q, K, V矩阵）的参数数量
attention_params_per_head = 3 * hidden_size  # Q, K, V 每个都是hidden_size
attention_params_per_layer = num_heads * attention_params_per_head

# 每个Transformer层的前馈网络参数量
ff_params_per_layer = 2 * (ff_size * hidden_size)  # 两个线性层

# Transformer层中还有LayerNorm和Dropout，但通常不计算在内（因为它们不增加可训练参数）
transformer_params_per_layer = attention_params_per_layer + ff_params_per_layer

# 所有Transformer层的总参数量
total_transformer_params = num_heads * transformer_params_per_layer

# 总参数量
total_params = embedding_params + total_transformer_params

# 打印结果
print(f"Embedding层参数量: {embedding_params}")
print(f"每层Transformer参数量: {transformer_params_per_layer}")
print(f"所有Transformer层总参数量: {total_transformer_params}")
print(f"总参数量: {total_params}")