"""
参数名参考config.json
参数：

1.embdding 层
word embedding: vocab * hidden
position embedding: max_position_embeddings * hidden
type(seg) embedding: type_vocab_size * hidden
layer norm: 
    weight: hidden
    bias: hidden

2.transformer 层
    2.1 attention
        q: hidden * hidden
        k: hidden * hidden
        v: hidden * hidden
        attention_out_w: hidden * hidden
        attention_out_b: hidden

    2.2 add & norm
        weight: hidden
        bias: hidden

    2.3 FFN
        w1: hidden * 4 hidden
        b1: 4 hidden
        w2: 4 hidden * hidden
        b2: hidden

    2.4 add & norm
        weight: hidden
        bias: hidden

    2.5 pooler
        weight: hidden * hidden
        bias: hidden

"""



def param_count(vocab_size,hidden_size,max_position_embeddings,num_hidden_layers,type_vocab_size):
    count = 0
    # 1.embdding 层
    param_dict = {}
    param_dict['word_embedding']  = vocab_size * hidden_size
    param_dict['pos_embedding'] = max_position_embeddings * hidden_size
    param_dict['type_embedding'] = type_vocab_size * hidden_size
    param_dict['emb_norm_w'] = hidden_size
    param_dict['emb_norm_b'] = hidden_size

    # 2.transformer
    param_dict['q_w'] =param_dict['q_b'] = param_dict['k_w'] = param_dict['k_b'] = param_dict['v_w'] = param_dict['v_b'] \
    = param_dict['attention_out_w'] = param_dict['attention_out_b'] = param_dict['attention_norm_w'] = param_dict['attention_norm_b'] = 0
    for _ in range(num_hidden_layers):
        param_dict['q_w'] = param_dict['q_w'] + hidden_size * hidden_size
        param_dict['q_b'] += hidden_size
        param_dict['k_w'] = param_dict['k_w'] + hidden_size * hidden_size 
        param_dict['k_b'] += hidden_size
        param_dict['v_w'] = param_dict['v_b'] + hidden_size * hidden_size
        param_dict['v_b'] += hidden_size
        param_dict['attention_out_w'] = param_dict['attention_out_w'] + hidden_size * hidden_size
        param_dict['attention_out_b'] = param_dict['attention_out_b'] + hidden_size
        param_dict['attention_norm_w'] = param_dict['attention_norm_w'] + hidden_size
        param_dict['attention_norm_b'] = param_dict['attention_norm_b'] +hidden_size

    # 3.FFN
    param_dict['ff_w1'] = hidden_size * 4 * hidden_size
    param_dict['ff_b1'] = 4 * hidden_size
    param_dict['ff_w2'] = 4 * hidden_size * hidden_size
    param_dict['ff_b2'] = hidden_size

    # 4.outpout_norm
    param_dict['output_norm_w'] = hidden_size
    param_dict['output_norm_b'] = hidden_size

    # 5. pooler
    param_dict['pooler_w'] = hidden_size * hidden_size
    param_dict['pooler_b'] = hidden_size

    count = 0
    for k,v in param_dict.items():
        print(k,':',v)
        count += v
        
    return count
    

vocab_size = 21128
hidden_size = 768
max_position_embeddings = 512
num_hidden_layers = 1  
type_vocab_size = 2
count = param_count(vocab_size,hidden_size,max_position_embeddings,num_hidden_layers,type_vocab_size)
print(count)
