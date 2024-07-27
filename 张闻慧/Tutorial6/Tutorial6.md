Given a config of a bert model as below:
{
  "architectures": [
    "BertForMaskedLM"
  ],
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
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "type_vocab_size": 2,
  "vocab_size": 21128
}

The number of parameters in the model is calculated as follows:

1. Embedding layers:

 1.1 segmentation token embedding: each token in the vocabulary is mapped to a vector of size hidden_size --- 22128*768 vocab_size *  hidden_size

 1.2 position embeddings: there are max_position_embeddings position embeddings, each of size hidden_size. 512 * 768 max_position_embeddings position * hidden_size

 1.3 token type embeddings: There are type_vocab_size token type embeddings, each of size hidden_size. 2 * 768  type_vocab_size * hidden_size

2.Transformer Encoder Layers  -- only calculated for one layer (in the end will be multiplied by num_hidden_layers)

 2.1 Self-Attention layer consists of:
	 2.1.1 attention weights (QKV) = 3*768*768  
	 2.1.2 output projection = num_attention_heads * head_dim * 768 = 768*768, where head_dim = hidden_size/number_attention_heads. 

 2.2 Feed-forward layer consists of:
 	 2.2.1 Intermediate projection = 768×3072 hidden_size * intermediate_size
 	 2.2.2 Output projection = 3072×768  intermediate_size * hidden_size


 2.3 Layer Norms: There are two layer norms in each layer (one before and one after the feed-forward network): =2×768×2  first 2 is for before and after and last 2 is for gamma and beta parameter. 


3. Pooler layer: feed-forward network applied to the first token's representation = 768×768+768



Total parameters: sum(Embedding layers) + num_hidden_layers * sum(Transformer Encoder Layers) + Pooler layer

 
 
















