# 计算bert的可训练参数数量
'''
  "attention_probs_dropout_prob": 0.1,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 1,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "type_vocab_size": 2,
  "vocab_size": 21128,
  "num_labels":18
'''


def calculate_parameters(vocab_size,
                         hidden_size,
                         max_position_embeddings,
                         type_vocab_size,
                         intermediate_size,
                         num_hidden_layers,
                         num_attention_heads):
    # Embedding Layer
    token_embeddings = vocab_size * hidden_size
    position_embeddings = max_position_embeddings * hidden_size
    token_type_embeddings = type_vocab_size * hidden_size
    total_embeddings = token_embeddings + position_embeddings + token_type_embeddings
    # print(total_embeddings)

    # Transformer Layers
    # qkv
    attention_params = 3 * (hidden_size * hidden_size)
    output_params = hidden_size * hidden_size
    # Intermediate dense layer & Output dense layer
    feed_forward_params = (hidden_size * intermediate_size) + (intermediate_size * hidden_size)
    layer_norm_params = 2 * hidden_size
    total_transformer = num_hidden_layers * (attention_params + output_params + feed_forward_params + layer_norm_params)
    # print(total_transformer)

    # Pooler Layer
    pooler_fc_size = hidden_size
    pooler_params = (hidden_size * pooler_fc_size) + (3 * pooler_fc_size * hidden_size)


    # Grand Total
    total_parameters = (total_embeddings + total_transformer + pooler_params)

    return total_parameters

vocab_size = 21128
hidden_size = 768
intermediate_size = 3072
max_position_embeddings = 512
num_hidden_layers = 1
num_attention_heads = 12
type_vocab_size = 2


total_params = calculate_parameters(vocab_size,
                                    hidden_size,
                                    max_position_embeddings,
                                    type_vocab_size,
                                    intermediate_size,
                                    num_hidden_layers,
                                    num_attention_heads)

print(f"总可训练参数量: {total_params}")