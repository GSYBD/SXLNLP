def count(vocab_size,hidden_size,max_position_embeddings,intermediate_size,):
    ebding_Size =vocab_size*hidden_size+hidden_size*2 +hidden_size*max_position_embeddings
    sef_attention_Size =3*hidden_size *hidden_size +3*hidden_size +hidden_size*hidden_size+hidden_size

    layerNorm_size =(hidden_size +hidden_size)
    feed_Size =hidden_size*intermediate_size +intermediate_size+intermediate_size*hidden_size +hidden_size
    pooler_size=hidden_size*hidden_size +hidden_size
    return ebding_Size +layerNorm_size+sef_attention_Size +layerNorm_size +feed_Size+layerNorm_size+pooler_size

print("total",count(21128,768,3072,3072))