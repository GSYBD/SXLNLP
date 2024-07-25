# 计算BERT中所有可训练参数的总数
# config.json中部分模型参数
attention_probs_dropout_prob = 0.1
hidden_dropout_prob = 0.1
hidden_size = 768
initializer_range = 0.02
intermediate_size = 3072
layer_norm_eps = 1e-12
max_position_embeddings = 512
num_attention_heads = 12
num_hidden_layers = 1
pad_token_id = 0
pooler_fc_size = 768
pooler_num_attention_heads = 12
pooler_num_fc_layers = 3
pooler_size_per_head = 128
type_vocab_size = 2
vocab_size = 21128
num_labels = 18

# 计算过程:
## embedding层参数量
para_num_word_embeddings = vocab_size * hidden_size
para_num_position_embeddings = 512 * hidden_size
para_num_segment_embeddings = 2 * hidden_size
total_para_num_embeddings = para_num_word_embeddings + para_num_position_embeddings + para_num_segment_embeddings

## self-attention层参数量
para_num_Q = hidden_size * (hidden_size / num_attention_heads) * num_attention_heads
para_num_K = hidden_size * (hidden_size / num_attention_heads) * num_attention_heads
para_num_V = hidden_size * (hidden_size / num_attention_heads) * num_attention_heads
total_para_num_self_attention = para_num_Q + para_num_K + para_num_V

## feed_forward层参数量
para_num_intermediate = hidden_size * intermediate_size
para_num_output = intermediate_size * hidden_size
total_para_num_feed_forward = para_num_intermediate + para_num_output

## pooler层参数量
para_num_pooler1 = 1 * pooler_size_per_head * pooler_num_attention_heads * pooler_fc_size
para_num_pooler2 = pooler_fc_size * pooler_size_per_head * pooler_num_attention_heads * pooler_fc_size
para_num_pooler3 = pooler_fc_size * pooler_size_per_head * pooler_num_attention_heads * pooler_fc_size
total_para_num_pooler = para_num_pooler1 + para_num_pooler2 + para_num_pooler3

# BERT参数量
para_num_bert = total_para_num_embeddings + total_para_num_self_attention + total_para_num_feed_forward + total_para_num_pooler
print(para_num_bert)