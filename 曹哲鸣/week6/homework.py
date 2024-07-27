from transformers import BertModel


# bert参数
attention_probs_dropout_prob = 0.1,
directionality = "bidi",
hidden_act = "gelu",
hidden_dropout_prob = 0.1,
hidden_size = 768,  #
initializer_range = 0.02,
intermediate_size = 3072,
layer_norm_eps = 1e-12,
max_position_embeddings = 512,
model_type = "bert",
num_attention_heads = 12,
num_hidden_layers = 12,
pad_token_id = 0,
pooler_fc_size = 768,
pooler_num_attention_heads = 12,
pooler_num_fc_layers = 3,
pooler_size_per_head = 128,
pooler_type = "first_token_transform",
type_vocab_size = 2,
vocab_size = 21128,
num_labels = 18


# 计算Embedding层参数
# Token Embedding层参数
token_embedding_params = vocab_size * hidden_size   # vocab_size字典大小，hidden_size每个字的维度
# Segment Embedding层参数
segment_embedding_params = 2 * hidden_size  # Segment Embedding数量为2
# Position Embedding层参数
position_embedding_params = max_position_embeddings * hidden_size
# Embedding层后的归一化层
layernormalization_embedding_params = hidden_size + hidden_size
# Embedding层可训练参数总数
embedding_parmas = token_embedding_params + segment_embedding_params + position_embedding_params + layernormalization_embedding_params


# Transformers层参数
# self-attention
# 计算K,V,Q时三个线性层的参数
transformers_linear_params = 3 * hidden_size *hidden_size + 3 * hidden_size     # hidden_size * hidden_size为线性层w参数，hidden_size为b参数
# 输出后再次经过一个线性层
output_linear_params = hidden_size * hidden_size + hidden_size
# 残差归一化层
layernormalization_attention_params = hidden_size + hidden_size


#Feed Forward
# Feed Forward共有两个线性层，一个在激活函数前，输出intermediate_size，一个在激活函数后，输入hidden_size
feedforward_params = (hidden_size * intermediate_size + intermediate_size) + (intermediate_size * hidden_size + hidden_size)
# 残差归一化层
layernormalization_feedforward_params = hidden_size + hidden_size

# Transformers层可训练参数总数,共有num_hidden_layers层Transformers层
transformers_params = (transformers_linear_params + output_linear_params + layernormalization_attention_params + feedforward_params + layernormalization_feedforward_params) * num_hidden_layers

# pooler层
pooler_params = hidden_size * hidden_size + hidden_size



#bert模型的可训练参数总数
total_params = embedding_parmas + transformers_params + pooler_params
