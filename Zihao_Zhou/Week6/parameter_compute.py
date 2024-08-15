hidden_size = 768
intermediate_size = 3072
max_position_embeddings = 512
num_hidden_layers = 1
pooler_fc_size = 768
vocab_size = 21128

# all_parameter = embedding_layer + encoder_layer + output_layer

# compute parameter of embedding_layer
embedding_normLayer_parameter = (vocab_size + 2 + max_position_embeddings) * hidden_size + hidden_size + hidden_size

# compute parameter of one encoder
attention_linear_normLayer_parameter = 3*(hidden_size**2+hidden_size)+hidden_size**2+hidden_size+hidden_size+hidden_size
up_down_linear_normLayer_parameter = 2*hidden_size*intermediate_size+intermediate_size+hidden_size+hidden_size+hidden_size

# compute parameter of encoder layer
encoder_parameter = num_hidden_layers*(attention_linear_normLayer_parameter + up_down_linear_normLayer_parameter)

# compute parameter of output layer
output_linear_parameter = hidden_size*pooler_fc_size+pooler_fc_size

# compute parameter of all
all_parameter = embedding_normLayer_parameter + encoder_parameter + output_linear_parameter
print(all_parameter)
print(24301056-all_parameter)