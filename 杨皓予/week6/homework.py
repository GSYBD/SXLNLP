# {
#   "architectures": [
#     "BertForMaskedLM"
#   ],
#   "attention_probs_dropout_prob": 0.1,
#   "directionality": "bidi",
#   "hidden_act": "gelu",
#   "hidden_dropout_prob": 0.1,
#   "hidden_size": 768,
#   "initializer_range": 0.02,
#   "intermediate_size": 3072,
#   "layer_norm_eps": 1e-12,
#   "max_position_embeddings": 512,
#   "model_type": "bert",
#   "num_attention_heads": 12,
#   "num_hidden_layers": 1,
#   "pad_token_id": 0,
#   "pooler_fc_size": 768,
#   "pooler_num_attention_heads": 12,
#   "pooler_num_fc_layers": 3,
#   "pooler_size_per_head": 128,
#   "pooler_type": "first_token_transform",
#   "type_vocab_size": 2,
#   "vocab_size": 21128,
#   "num_labels":18
# }

hidden_size = 768
vocab_size = 21128
num_segment = 2
max_position_embeddings = 512
# feedForward层中过激活函数前的中间线性层的output_size,即intermediate_size，bert-base-chinese的intermediate_size为3072
intermediate_size = 3072
# bert模型中transformer层的数量,设置为1
num_hidden_layers = 1
# 计算bert模型的可训练参数量
# 1.计算embedding层的可训练参数量
token_embedding_param = vocab_size * hidden_size  # 词嵌入层的可训练参数量
segment_embedding_param = num_segment * hidden_size  # 句子嵌入层的可训练参数量
position_embedding_param = max_position_embeddings * hidden_size  # 位置嵌入层的可训练参数量

# 2.计算token_embedding_param、segment_embedding_param、position_embedding_param加和之后的规划层embeddingLayerNorm(
# XtokenEmbedding + XsegmentEmbedding + XpositionEmbedding)的可训练参数量
embedding_layer_norm_param = hidden_size + hidden_size  # 第一个hidden_size代表embeddingLayerNorm的w的可训练参数数量，第二个hidden_size
# 代表embeddingLayerNorm的b的可训练参数数量

# 3.计算transformer层中self-attention中的可训练参数量
self_attention_param = hidden_size * hidden_size * 3 + hidden_size * 3  # hidden_size * hidden_size *
# 3代表三个线性层的w的可训练参数数量，hidden_size * 3代表三个线性层的b的可训练参数数量
# 4.计算self-attention-output层中的可训练参数量
self_attention_output_param = hidden_size * hidden_size + hidden_size  # hidden_size *
# hidden_size代表线性层的w的可训练参数数量，hidden_size代表线性层的b的可训练参数数量
# 5.计算self-attention-output之后的残差规划层attentionLayerNorm(Xembedding+ Xattention)的可训练参数量
attention_layer_norm_param = hidden_size + hidden_size  # 第一个hidden_size代表attentionLayerNorm
# 的w的可训练参数数量，第二个hidden_size代表attentionLayerNorm的b的可训练参数数量
# 6.计算feedForward层中的可训练参数量
feed_forward_param = (intermediate_size * hidden_size + intermediate_size) + (hidden_size * intermediate_size + hidden_size)  # (intermediate_size * hidden_size + intermediate_size)代表线性层的w的可训练参数数量，
# (hidden_size * intermediate_size + hidden_size)代表线性层的b的可训练参数数量

# 7.计算feedForward层之后的残差规划层feedForwardLayerNorm(Xattention + XfeedForward)的可训练参数量
feed_forward_layer_norm_param = hidden_size + hidden_size  # 第一个hidden_size 代表feedForwardLayerNorm的w的可训练参数数量，第二个hidden_size代表feedForwardLayerNorm的b的可训练参数数量
# 8.pooler层的可训练参数量
pooler_param = hidden_size * hidden_size + hidden_size  # hidden_size * hidden_size代表线性层的w的可训练参数数量，hidden_size代表线性层的b的可训练参数数量
# 9.bert模型的总可训练参数量
total_param = token_embedding_param + segment_embedding_param + position_embedding_param + embedding_layer_norm_param + \
              (
                          self_attention_param + self_attention_output_param + attention_layer_norm_param + feed_forward_param + feed_forward_layer_norm_param) * num_hidden_layers \
              + pooler_param

print("bert-base-chinese模型的总参数量为：", total_param)