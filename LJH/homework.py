from transformers import BertModel

bert = BertModel.from_pretrained(r"C:\Users\19107\Desktop\bd\NLP\第六周 预训练模型\bert-base-chinese", return_dict=False,output_hidden_states=True)
state_dict = bert.state_dict()
for k,v in state_dict.items():
    print(k,v.shape)

'''
x12
embedding:
embeddings.word_embeddings.weight (vocab_size, hidden_size)
embeddings.position_embeddings.weight (max_position_embeddings, hidden_size)
embeddings.token_type_embeddings.weight (type_vocab_size, hidden_size)
embeddings.LayerNorm.weight (hidden_size,)
self-attention:
encoder.layer.0.attention.self.query.weight (hidden_size, hidden_size)
encoder.layer.0.attention.self.key.weight (hidden_size, hidden_size)
encoder.layer.0.attention.self.value.weight (hidden_size, hidden_size)
encoder.layer.0.attention.output.dense.weight (hidden_size, hidden_size)
residual:
encoder.layer.0.attention.output.LayerNorm.weight (hidden_size,)
feed:
encoder.layer.0.intermediate.dense.weight (4*hidden_size, hidden_size)
encoder.layer.0.output.dense.weight (hidden_size, 4*hidden_size)
encoder.layer.0.output.LayerNorm.weight (hidden_size,)
pooler:
pooler.dense.weight (hidden_size, hidden_size)
'''