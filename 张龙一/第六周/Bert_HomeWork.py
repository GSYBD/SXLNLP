def calculate_bert_base_params(hidden_size=768, num_hidden_layers=12, vocab_size=21128, max_position_embeddings=512,
                               num_attention_heads=12, intermediate_size=3072):
    """
    计算BERT-Base模型的可训练参数数量。

    参数:
    - hidden_size: 隐藏层大小。
    - num_hidden_layers: Encoder层的数量。
    - vocab_size: 词表大小。
    - max_position_embeddings: 位置嵌入的最大长度。
    - num_attention_heads: Multi-Head Attention中的head数量。
    - intermediate_size: Feed Forward层的中间层维度。

    返回:
    - 总的可训练参数数量。
    """
    # Embedding层
    embedding_params = (vocab_size + 2 + max_position_embeddings) * hidden_size + hidden_size * 2  # Token, Segment, Position, 2*LayerNorm

    # Encoder层
    multi_head_attention_params_per_layer = 12 * 3 * (hidden_size * (hidden_size // num_attention_heads) + (hidden_size // num_attention_heads)) + hidden_size * hidden_size +  hidden_size # QKV + output projection
    feed_forward_params_per_layer = hidden_size * intermediate_size + intermediate_size + intermediate_size * hidden_size + hidden_size# two fully-connected layers
    layer_norm_params_per_encoder = 2 * hidden_size * 2  # two LayerNorm per encoder (one after attention, one after feed-forward)
    encoder_params = num_hidden_layers * (multi_head_attention_params_per_layer + feed_forward_params_per_layer + layer_norm_params_per_encoder)

    # Pooling层通常不计算在内，因为它可能不参与训练（除非有特定的下游任务）
    # 但为了完整性，我们可以计算它（如果需要的话）
    pooling_params = hidden_size * hidden_size + hidden_size

    # 总参数
    total_params = embedding_params + encoder_params + pooling_params

    return total_params

custom_params = calculate_bert_base_params(hidden_size=768, num_hidden_layers=1)
print(f"Total trainable parameters with custom settings: {custom_params:,} ≈ {custom_params / 1e6:.2f}M")
