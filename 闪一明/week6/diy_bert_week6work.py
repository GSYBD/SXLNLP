import torch
import math
import numpy as np
from transformers import BertModel

# 通过手动矩阵运算实现 Bert 结构
bert = BertModel.from_pretrained(r"F:\ai学习录播\第六周 预训练模型\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()


class DiyBert:
    def __init__(self, state_dict):
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 1  # 注意这里的层数要跟预训练 config.json 文件中的模型层数一致
        self.load_weights(state_dict)

    def load_weights(self, state_dict):
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
        self.transformer_weights = []
        for i in range(self.num_layers):
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy()
            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy()
            k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy()
            k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy()
            v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy()
            v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy()
            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy()
            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy()
            attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy()
            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy()
            intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy()
            intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy()
            output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy()
            output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy()
            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy()
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()
            self.transformer_weights.append(
                [q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                 attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                 output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()

    # 计算总参数量
    def count_parameters(self):
        total_params = 0
        # Embedding部分
        total_params += np.prod(self.word_embeddings.shape)
        total_params += np.prod(self.position_embeddings.shape)
        total_params += np.prod(self.token_type_embeddings.shape)
        total_params += np.prod(self.embeddings_layer_norm_weight.shape)
        total_params += np.prod(self.embeddings_layer_norm_bias.shape)

        # Transformer部分
        for layer in self.transformer_weights:
            for weight in layer:
                total_params += np.prod(weight.shape)

        # Pooler层
        total_params += np.prod(self.pooler_dense_weight.shape)
        total_params += np.prod(self.pooler_dense_bias.shape)

        return total_params


# 实例化模型并计算参数量
db = DiyBert(state_dict)
print("Total trainable parameters:", db.count_parameters())
