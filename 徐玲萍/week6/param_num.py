bert参数计算：
总数：2430 1056

分步：
768 * (21128 + 512 + 2 + 2) = 1662 2592

(768 * 768 + 768)(3 + 1) + 768 * 2 + (768 * 4)(768+1+768) + 768 * 3 = 7087872
num_hidden_layers为 1 7087872* 1 = 708 7872

768 * 768 + 768 = 59 0592

1662 2592 + 708 7872 + 59 0592 = 2430 1056

计算明细
embedding部分 hidden_size * (vocab_size + max_position_embeddings + type_vocab_size + 2)
word_embeddings token词向量 vocab_size * hidden_size 21128 * 768
position_embeddings: 定位词向量 max_position_embeddings * hidden_size 512 * 768
token_type_embeddings：上下句向量 type_vocab_size * hidden_size 2 * 768
embeddings_layer_norm_weight：向量归一化层，也就是一个线性层，帮助网络更好地学习和收敛。 hidden_size，
embeddings_layer_norm_bias ：向量归一化层的b hidden_size


transformer_weights：transforms层的参数，这要看有几层 num_hidden_layers
self-attention层
(hidden_size * hidden_size + hidden_size) * (3+1)  + hidden_size * 2 + (4*hidden_size)(hidden_size+1+hidden_size) +hidden_size*3
q_w,q_b,k_w,k_b,v_w,v_b: q k v 三个线性网络参数
         w都是 hidden_size * hidden_size 768* 768  b都是hidden_size 768
attention_output_weight：将词向量过一个线性转换层，转成需要的维度 hidden_size*hidden_size 768* 768
attention_output_bias： 上面线性层的b hidden_size 768
attention_layer_norm_w：归一化层 减少优化过程中的内部协变量偏移和梯度消失等问题，从而帮助网络更好地训练和收敛。
attention_layer_norm_b：这两个都是 hidden_size 768，TODO 为什么是hidden_size？怎么乘呢

encoder:编码器层
intermediate_weight：将768维 映射为4倍的高维，因为Feedforward 前馈层，需要高维来捕捉跟表示输入数据特征
           形状 hidden_size * hidden_size*4  768*3072
intermediate_bias：hidden_size*4  3072

Feedforward：前馈层
output_weight ：线性转换层，将 intermediate的高维，映射回隐藏层维度
        形状 hidden_size*4 * hidden_size 3072*768
output_bias：hidden_size 768
ff_layer_norm_w：归一化层 hidden_size  768
ff_layer_norm_b：hidden_size  768


pooler层:池化层
hidden_size * hidden_size + hidden_size
pooler_dense_weight: 线性转换的池化 hidden_size * hidden_size 768* 768
pooler_dense_bias：hidden_size 768
