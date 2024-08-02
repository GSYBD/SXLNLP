from transformers import BertModel

# 初始化BERT模型
model = BertModel.from_pretrained(r"E:\BaiduNetdiskDownload\bert-base-chinese", return_dict=False)

# 获取模型配置
config = model.config
d_model = config.hidden_size  # 嵌入维度
vocab_size = config.vocab_size  # 词汇表大小
max_position_embeddings = config.max_position_embeddings  # 最大位置嵌入

# 计算Embedding层参数量
token_embeddings_params = vocab_size * d_model
position_embeddings_params = max_position_embeddings * d_model
segment_embeddings_params = 2 * d_model  # 2个段的嵌入

# 添加标准化的权重和偏置
embedding_layer_norm_params = d_model   # 权重和偏置

embedding_params = (token_embeddings_params + position_embeddings_params +
                    segment_embeddings_params + 2*embedding_layer_norm_params)

# 计算每层的参数量
layer_params = 0
num_layers = config.num_hidden_layers
print(num_layers,d_model)
for _ in range(num_layers):
    # Self-Attention参数量
    attention_params = 3* (d_model * d_model) +3 * d_model # Q, K, V, 输出层
    output_params=d_model * d_model+d_model
    # Layer Norm参数量
    layer_norm_params = 4* d_model
    # Feed Forward参数量
    feed_forward_params = (d_model * 4 * d_model) + (4 * d_model) + (4 * d_model * d_model) + d_model  # 两层


    layer_params += (attention_params + feed_forward_params + layer_norm_params
                     +output_params)

# 计算输出层参数量
output_layer_params = d_model * d_model+d_model

# 总参数量
total_params = embedding_params + layer_params  + output_layer_params

# 打印每层的形状和参数量
print(f"Embedding层参数量: {embedding_params}")
print(f"每层参数量: {layer_params}")
print(f"输出层参数量: {output_layer_params}")
print(f"\n总可训练参数量: {total_params}")
