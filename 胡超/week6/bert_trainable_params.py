# -*- coding: utf-8 -*-
"""
author: Chris Hu
date: 2024/7/25
desc:
sample
"""

import json


def count_trainable_params(config_path):
    with open(config_path, 'r') as f:
        content = f.read()
        bert_params = json.loads(content)

    hidden_size = bert_params['hidden_size']
    intermediate_size = bert_params['intermediate_size']
    max_position_embeddings = bert_params['max_position_embeddings']
    num_hidden_layers = bert_params['num_hidden_layers']
    type_vocab_size = bert_params['type_vocab_size']
    vocab_size = bert_params['vocab_size']

    # 1. embedding layer
    # count the number of trainable parameters in embedding
    # i.e. params of (token embeddings + segment embeddings + position embeddings + embedding normalization)
    token_embedding_params = hidden_size * vocab_size
    seg_ment_embedding_params = hidden_size * type_vocab_size
    position_embedding_params = hidden_size * max_position_embeddings
    embedding_norm_params = hidden_size + hidden_size
    embedding_layer_params = token_embedding_params + seg_ment_embedding_params + position_embedding_params + \
                             embedding_norm_params

    # 2. transformer encoder layer
    # count the number of trainable parameters in transformer encoder layer
    # i.e. params of (multi-head attention + feed-forward network)
    # params model: w + b
    # Simplified QKV calculation due to 3 * num_attention_heads * (hidden_size / num_attention_heads) * hidden_size
    # is just 3 * hidden_size * hidden_size + hidden_size
    self_attention_qkv_params = 3 * hidden_size * hidden_size + hidden_size * 3
    self_attention_output_params = hidden_size * hidden_size + hidden_size
    self_attention_norm_params = hidden_size + hidden_size
    # params of linear layer before hidden_act(i.e. gelu)
    feed_forword_params_before_hidden_act = hidden_size * intermediate_size + intermediate_size
    # params of linear layer after hidden_act(i.e. gelu)
    feed_forword_params_after_hidden_act = intermediate_size * hidden_size + hidden_size
    feed_forward_norm_params = hidden_size + hidden_size
    transformer_encoder_layer_params = (self_attention_qkv_params + self_attention_output_params +
                                        self_attention_norm_params + feed_forword_params_before_hidden_act +
                                        feed_forword_params_after_hidden_act +
                                        feed_forward_norm_params) * num_hidden_layers

    # 3. pooler layer
    # full connected layer
    # params model: w + b
    pooler_layer_params = hidden_size * hidden_size + hidden_size

    return embedding_layer_params + transformer_encoder_layer_params + pooler_layer_params


if __name__ == '__main__':
    trainable_params_count = count_trainable_params(r'./config.json')
    print("基于当前的Bert配置文件config.json，Bert模型的可训练参数个数为：", trainable_params_count)
