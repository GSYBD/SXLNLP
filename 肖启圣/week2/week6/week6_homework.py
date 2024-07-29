import json

file_path = "config.json"

h = 2
vocab = json.load(open(file_path, "r", encoding="utf8"))

# embedding layer
word_embeddings = vocab["vocab_size"] * vocab["hidden_size"]
token_type_embeddings = h * vocab["hidden_size"]
position_embeddings = vocab["max_position_embeddings"] * vocab["hidden_size"]
embeddings_layer_norm_weight = vocab["hidden_size"]
embeddings_layer_norm_bias =  vocab["hidden_size"]

total_embedding_layer = word_embeddings + token_type_embeddings + position_embeddings + embeddings_layer_norm_weight + embeddings_layer_norm_bias

# self.attention.layer
q_w = vocab["hidden_size"] * vocab["hidden_size"]
q_b = vocab["hidden_size"]
k_w =vocab["hidden_size"] * vocab["hidden_size"]
k_b =vocab["hidden_size"]
v_w =vocab["hidden_size"] * vocab["hidden_size"]
v_b = vocab["hidden_size"]
attention_output_weight = vocab["hidden_size"] * vocab["hidden_size"]
attention_output_bias = vocab["hidden_size"]
attention_layer_norm_w = vocab["hidden_size"]
attention_layer_norm_b = vocab["hidden_size"]
total_attention_layer = q_w + q_b + k_w+k_b+v_w+v_b+attention_output_weight+attention_output_bias+attention_layer_norm_w+attention_layer_norm_b

# feed.forward.layer
intermediate_weight = vocab["hidden_size"] * vocab["intermediate_size"]
intermediate_bias = vocab["intermediate_size"]
output_weight = vocab["intermediate_size"] * vocab["hidden_size"]
output_bias = vocab["hidden_size"]
ff_layer_norm_w = vocab["hidden_size"]
ff_layer_norm_b = vocab["hidden_size"]
total_feed_forward = intermediate_weight+intermediate_bias+output_weight+output_bias+ff_layer_norm_w+ff_layer_norm_b

# feed.forward.layer
pool_fc_w = vocab["hidden_size"] * vocab["hidden_size"]
pool_fc_b =vocab["hidden_size"]
pool_fc = pool_fc_w + pool_fc_b

total_parameter = total_embedding_layer +(total_attention_layer +total_feed_forward + pool_fc)*vocab["num_hidden_layers"]
print(total_parameter)