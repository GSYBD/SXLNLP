import torch
import math
import numpy as np
from transformers import BertModel

'''
通过手动矩阵运算实现Bert结构
模型文件下载 https://huggingface.co/models
'''

# 加载预训练模型
bert = BertModel.from_pretrained("bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
bert.eval()

# 假想一个4个字的句子
x = np.array([2450, 15486, 102, 2110])   # 假想成4个字的句子
torch_x = torch.LongTensor([x])          # pytorch形式输入

# 得到模型的输出
sequence_output, pooler_output = bert(torch_x)
print(sequence_output.shape, pooler_output.shape)
# print(sequence_output, pooler_output)

print(bert.state_dict().keys())  # 查看所有的权值矩阵名称

# softmax归一化
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# gelu激活函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))

class DiyBert:
    # 将预训练好的整个权重字典输入进来
    def __init__(self, state_dict):
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 1  # 注意这里的层数要跟预训练config.json文件中的模型层数一致
        self.load_weights(state_dict)

    def load_weights(self, state_dict):
        # embedding部分
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
        self.transformer_weights = []
        # transformer部分，有多层
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
            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                                             attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                                             output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
        # pooler层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()

    # bert embedding，使用3层叠加，在经过一个Layer norm层
    def embedding_forward(self, x):
        we = self.get_embedding(self.word_embeddings, x)
        pe = self.get_embedding(self.position_embeddings, np.array(list(range(len(x)))))
        te = self.get_embedding(self.token_type_embeddings, np.array([0] * len(x)))
        embedding = we + pe + te
        embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)
        return embedding

    # embedding层实际上相当于按index索引，或理解为onehot输入乘以embedding矩阵
    def get_embedding(self, embedding_matrix, x):
        return np.array([embedding_matrix[index] for index in x])

    # 执行全部的transformer层计算
    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x, i)
        return x

    # 执行单层transformer层计算
    def single_transformer_layer_forward(self, x, layer_index):
        weights = self.transformer_weights[layer_index]
        q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias, attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias, output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b = weights
        attention_output = self.self_attention(x, q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias, self.num_attention_heads, self.hidden_size)
        x = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)
        feed_forward_x = self.feed_forward(x, intermediate_weight, intermediate_bias, output_weight, output_bias)
        x = self.layer_norm(x + feed_forward_x, ff_layer_norm_w, ff_layer_norm_b)
        return x

    # self attention的计算
    def self_attention(self, x, q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias, num_attention_heads, hidden_size):
        q = np.dot(x, q_w.T) + q_b
        k = np.dot(x, k_w.T) + k_b
        v = np.dot(x, v_w.T) + v_b
        attention_head_size = int(hidden_size / num_attention_heads)
        q = self.transpose_for_scores(q, attention_head_size, num_attention_heads)
        k = self.transpose_for_scores(k, attention_head_size, num_attention_heads)
        v = self.transpose_for_scores(v, attention_head_size, num_attention_heads)
        qk = np.matmul(q, k.swapaxes(1, 2))
        qk /= np.sqrt(attention_head_size)
        qk = softmax(qk)
        qkv = np.matmul(qk, v)
        qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)
        attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias
        return attention

    # 多头机制
    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        max_len, hidden_size = x.shape
        x = x.reshape(max_len, num_attention_heads, attention_head_size)
        x = x.swapaxes(1, 0)
        return x

    # 前馈网络的计算
    def feed_forward(self, x, intermediate_weight, intermediate_bias, output_weight, output_bias):
        x = np.dot(x, intermediate_weight.T) + intermediate_bias
        x = gelu(x)
        x = np.dot(x, output_weight.T) + output_bias
        return x

    # 归一化层
    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b
        return x

    # 链接[cls] token的输出层
    def pooler_output_layer(self, x):
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        x = np.tanh(x)
        return x

    # 最终输出
    def forward(self, x):
        x = self.embedding_forward(x)
        sequence_output = self.all_transformer_layer_forward(x)
        pooler_output = self.pooler_output_layer(sequence_output[0])
        return sequence_output, pooler_output

# 自制
db = DiyBert(state_dict)
diy_sequence_output, diy_pooler_output = db.forward(x)

# torch
torch_sequence_output, torch_pooler_output = bert(torch_x)

print(diy_sequence_output)
print(torch_sequence_output)

# 计算并输出BERT模型的参数数量
V = 30522  # 词汇表大小
P = 512    # 最大序列长度
T = 2      # token类型数量
H = 768    # 隐藏层维度
L = 12     # Transformer层数量

embedding_params = V * H + P * H + T * H + 2 * H
transformer_layer_params = 12 * H**2 + 13 * H
total_transformer_params = L * transformer_layer_params
pooler_params = H**2 + H
total_params = embedding_params + total_transformer_params + pooler_params

print(f"Embedding parameters: {V} * {H} + {P} * {H} + {T} * {H} + 2 * {H} = {embedding_params}")
print(f"Transformer layer parameters (per layer): 12 * {H}^2 + 13 * {H} = {transformer_layer_params}")
print(f"Total transformer parameters: {L} * ({transformer_layer_params}) = {total_transformer_params}")
print(f"Pooler parameters: {H}^2 + {H} = {pooler_params}")
print(f"Total parameters in BERT model: {total_params}")
