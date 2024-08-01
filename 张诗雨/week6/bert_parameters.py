def bert_parameters():
    vocab_size = 30522
    hidden_size = 768
    max_position_embeddings = 512
    type_vocab_size = 2
    num_attention_heads = 12
    intermediate_size = 3072
    num_hidden_layers = 12

    # 1. Embedding层参数
    token_embeddings = vocab_size * hidden_size
    segment_embeddings = type_vocab_size * hidden_size
    position_embeddings = max_position_embeddings * hidden_size
    total_embeddings = token_embeddings + segment_embeddings + position_embeddings

    # 2. Self-Attention层参数
    QKV_weight = 3 * hidden_size * hidden_size
    QKV_bias = 3 * hidden_size
    attention_output_weight = hidden_size * hidden_size
    attention_output_bias = hidden_size
    attention_layer_norm = 2 * hidden_size
    total_attention = QKV_weight + QKV_bias + attention_output_weight + attention_output_bias + attention_layer_norm

    # 3. Feed Forward层参数
    feed_forward_weight_1 = hidden_size * intermediate_size
    feed_forward_bias_1 = intermediate_size
    feed_forward_weight_2 = intermediate_size * hidden_size
    feed_forward_bias_2 = hidden_size
    feed_forward_layer_norm = 2 * hidden_size
    total_feed_forward = feed_forward_weight_1 + feed_forward_bias_1 + feed_forward_weight_2 + feed_forward_bias_2 + feed_forward_layer_norm

    total_bert_parameters = total_embeddings + num_hidden_layers * (total_attention + total_feed_forward)
    print(total_bert_parameters)


bert_parameters()