import torch
import math
import numpy as np
from transformers import BertModel
# 根据bert-base-chinese的参数，hidden_size = 768 vocab_size=21128
hidden_size = 768
vocab_size = 21128
# bert模型 固定num_segment 为2，max_position_embeddings 为512
num_segment = 2
max_position_embeddings = 512
# feedForward层中过激活函数前的中间线性层的output_size,即intermediate_size，bert-base-chinese的intermediate_size为3072
intermediate_size = 3072
# bert模型中transformer层的数量,设置为1
num_hidden_layers = 1
# 计算bert模型的可训练参数量
# 1.计算embedding层的可训练参数量
token_embedding_param = hidden_size * vocab_size # tokenEmbedding的参数量 = hidden_size * vocab_size
segment_embedding_param = hidden_size * num_segment  # segmentEmbedding的参数量 = hidden_size * num_segment
position_embedding_param = hidden_size * max_position_embeddings  # positionEmbedding的参数量 = hidden_size *
# max_position_embeddings

# 2.计算token_embedding_param、segment_embedding_param、position_embedding_param加和之后的规划层embeddingLayerNorm(
# XtokenEmbedding + XsegmentEmbedding + XpositionEmbedding)的可训练参数量
embedding_layer_norm_param = hidden_size + hidden_size  # 第一个hidden_size 代表embeddingLayerNorm
# 的w的可训练参数数量，第二个hidden_size代表embeddingLayerNorm的b的可训练参数数量

# 3.计算transformer层中self-attention中的可训练参数量
self_attention_param = hidden_size * hidden_size * 3 + hidden_size * 3  # hidden_size * hidden_size *
# 3代表三个线性层的w的可训练参数数量，hidden_size * 3代表三个线性层的b的可训练参数数量
# 4.计算self-attention-output层中的可训练参数量
self_attention_output_param = hidden_size * hidden_size + hidden_size  # hidden_size *
# hidden_size代表线性层的w的可训练参数数量，hidden_size代表线性层的b的可训练参数数量
# 5.计算self-attention-output之后的残差规划层attentionLayerNorm(Xembedding+ Xattention)的可训练参数量
attention_layer_norm_param = hidden_size + hidden_size  # 第一个hidden_size代表attentionLayerNorm
# 的w的可训练参数数量，第二个hidden_size代表attentionLayerNorm的b的可训练参数数量
# 6.计算feedForward层中的可训练参数量
feed_forward_param = (hidden_size * intermediate_size + intermediate_size) + \
            (intermediate_size * hidden_size + hidden_size)  # hidden_size * intermediate_size +
# intermediate_size过激活函数前的中间线性层的可训练参数量，其中hidden_size *
# intermediate_size是w的参数量，intermediate_size是b的参数量，intermediate_size * hidden_size + hidden_size
# 是过了激活函数后的线性层的可训练参数量，其中intermediate_size * hidden_size是w的参数量，hidden_size是b的参数量

# 7.计算feedForward层之后的残差规划层feedForwardLayerNorm(Xattention + XfeedForward)的可训练参数量
feed_forward_layer_norm_param = hidden_size + hidden_size  # 第一个hidden_size 代表feedForwardLayerNorm的w的可训练参数数量，第二个hidden_size代表feedForwardLayerNorm的b的可训练参数数量
# 8.pooler层的可训练参数量
pooler_param = hidden_size * hidden_size + hidden_size  # hidden_size * hidden_size代表线性层的w的可训练参数数量，hidden_size代表线性层的b的可训练参数数量
# 9.bert模型的总可训练参数量
total_param = token_embedding_param + segment_embedding_param + position_embedding_param + embedding_layer_norm_param + \
            (self_attention_param + self_attention_output_param + attention_layer_norm_param + feed_forward_param + feed_forward_layer_norm_param) * num_hidden_layers \
            + pooler_param

if __name__ == "__main__":
    bert = BertModel.from_pretrained(r"E:\ai课程\八斗精品班\week6 语言模型和预训练\bert-base-chinese", return_dict=False)
    print(f"手动计算bert模型总可训练参数量:%d" % total_param)
    print(f"bert-base-chinese模型总可训练参数量:%d" % bert.num_parameters())
    # state_dict = bert.state_dict()
    # word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()
    # print(f"word_embeddings:{word_embeddings.shape}" )
    # position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()
    # print(f"position_embeddings:{position_embeddings.shape}")
    # token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
    # print(f"token_type_embeddings:{token_type_embeddings.shape}")
    # embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
    # print(f"embeddings_layer_norm_weight:{embeddings_layer_norm_weight.shape}")
    # embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
    # print(f"embeddings_layer_norm_bias:{embeddings_layer_norm_bias.shape}")
    # q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % 0].numpy()
    # print(f"q_w:{q_w.shape}")
    # q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % 0].numpy()
    # print(f"q_b:{q_b.shape}")
    # k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % 0].numpy()
    # print(f"k_w:{k_w.shape}")
    # k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % 0].numpy()
    # print(f"k_b:{k_b.shape}")
    # v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % 0].numpy()
    # print(f"v_w:{v_w.shape}")
    # v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % 0].numpy()
    # print(f"v_b:{v_b.shape}")
    # attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % 0].numpy()
    # print(f"attention_output_weight:{attention_output_weight.shape}")
    # attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % 0].numpy()
    # print(f"attention_output_bias:{attention_output_bias.shape}")
    # attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % 0].numpy()
    # print(f"attention_layer_norm_w:{attention_layer_norm_w.shape}")
    # attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % 0].numpy()
    # print(f"attention_layer_norm_b:{attention_layer_norm_b.shape}")
    # intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % 0].numpy()
    # print(f"intermediate_weight:{intermediate_weight.shape}")
    # intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % 0].numpy()
    # print(f"intermediate_bias:{intermediate_bias.shape}")
    # output_weight = state_dict["encoder.layer.%d.output.dense.weight" % 0].numpy()
    # print(f"output_weight:{output_weight.shape}")
    # output_bias = state_dict["encoder.layer.%d.output.dense.bias" % 0].numpy()
    # print(f"output_bias:{output_bias.shape}")
    # ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % 0].numpy()
    # print(f"ff_layer_norm_w:{ff_layer_norm_w.shape}")
    # ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % 0].numpy()
    # print(f"ff_layer_norm_b:{ff_layer_norm_b.shape}")
    # pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
    # print(f"pooler_dense_weight:{pooler_dense_weight.shape}")
    # pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()
    # print(f"pooler_dense_bias:{pooler_dense_bias.shape}")