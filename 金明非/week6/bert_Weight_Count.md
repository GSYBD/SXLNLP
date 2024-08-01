bert:
    embedding:
        token_embedding: [vocab_size, hidden_size]
        segment_embedding: [2, hidden_size]
        position_embedding: [max_position_embeddings, hidden_size]
        
        embedding 总参数 = (vocab_size + 2 + max_position_embeddings) * hidden_size


    transformer:
        for i in num_hidden_layers:
            self-attention:
                Q_w: [hidden_size, hidden_size]
                K_W: [hidden_size, hidden_size]
                v_w: [hidden_size, hidden_size]
            
                output = Liner(attention(Q,K,V))
                => Liner: [hidden_size, hidden_size]
            单层self-attention的参数 =  Q,K,V参数 + output参数
                                    = (3 * hidden_size * hidden_size) + (hidden_size * hidden_size)
                                    = 4 * hidden_size


            Feed Forward:
                output = Liner2(gelu(Liner1(x))) =>
                Liner1: [hidden_size, intermediate_size]
                Liner2: [intermediate_size, hidden_size]
            单层Feed Forward的参数 =  2 * hidden_size * intermediate_size
            
        transformer的总参数 = 层数 * 单层self-attention参数 * 单层Feed Forward 参数
                           = num_hidden_layers * (hidden_size * hidden_size * 3) * (2 * hidden_size * intermediate_size)

    bert的总参数 = embedding参数 + transformer参数
                = ((vocab_size + 2 + max_position_embeddings) * hidden_size) + (num_hidden_layers * (4 * hidden_size * hidden_size) * (2 * hidden_size * intermediate_size))

