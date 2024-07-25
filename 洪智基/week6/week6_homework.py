#coding:utf8

'''
已知bert模型配置如下：
{
  "architectures": ["BertForMaskedLM"],     列出模型支持的架构类型，这里是BertForMaskedLM，表示这是一个用于掩码语言模型任务的BERT模型。
  "attention_probs_dropout_prob": 0.1,      注意力权重上的dropout概率，用于防止过拟合，这里是0.1。
  "directionality": "bidi",                 表示模型是双向的（bidi），即同时考虑上下文信息。
  "hidden_act": "gelu",                     隐藏层激活函数，这里是gelu（高斯误差线性单元），一种非线性激活函数。
  "hidden_dropout_prob": 0.1,               隐藏层上的dropout概率，同样是0.1，用于防止过拟合。
  "hidden_size": 768,                       隐藏层的大小，即Transformer编码器中每个层的嵌入大小，这里是768。
  "initializer_range": 0.02,                初始化参数的范围，这里是0.02，用于权重初始化。
  "intermediate_size": 3072,                中间层（即前馈网络层）的大小，这里是3072，比隐藏层大，用于增加模型的非线性。
  "layer_norm_eps": 1e-12,                  层归一化中的epsilon值，用于防止除以零的错误，这里是1e-12。
  "max_position_embeddings": 512,           模型能处理的最大位置嵌入数，这里是512，表示序列的最大长度。
  "model_type": "bert",                     模型的类型，这里是bert。
  "num_attention_heads": 12,                注意力机制中头的数量，这里是12，多头注意力机制允许模型同时从不同的子空间学习信息。
  "num_hidden_layers": 12,                  隐藏层的数量，即Transformer编码器的层数，这里是12。
  "pad_token_id": 0,                        填充（padding）标记的ID，这里是0，用于处理不等长序列。
  "pooler_fc_size": 768,                    池化层全连接层的大小，这里是768，用于生成整个序列的表示。
  "pooler_num_attention_heads": 12,         池化层中多头注意力机制的头数，这里是12，但通常池化层不使用多头注意力。
  "pooler_num_fc_layers": 3,                池化层中全连接层的数量，这里是3。
  "pooler_size_per_head": 128,              池化层中每个注意力头的大小，这里是128，但通常此参数与池化层不直接相关。
  "pooler_type": "first_token_transform",   池化层的类型，这里是first_token_transform，意味着使用序列的第一个标记（通常是[CLS]标记）的变换作为整个序列的表示。
  "type_vocab_size": 2,                     类型词汇表的大小，这里是2，通常用于区分不同的句子（在BERT的NSP任务中）。
  "vocab_size": 21128,                      词汇表的大小，即模型能识别的词（或标记）的数量，这里是21128。
  "num_labels":18                           任务中的标签数量，这里是18，表示该模型可能被用于一个具有18个类别的分类任务。
}
'''

#将模型配置存入list
ModelParameter_Bert ={
  "architectures": ["BertForMaskedLM"],
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
  "vocab_size": 21128,
  "num_labels":18
}

#计算bert模型可训练参数
# 1. 嵌入层（Embeddings）
# 词汇嵌入（Token Embeddings）：
# vocab_size * hidden_size = 21128 * 768
Token = ModelParameter_Bert["vocab_size"] * ModelParameter_Bert["hidden_size"]

# 位置嵌入（Position Embeddings）：
# max_position_embeddings * hidden_size = 512 * 768
Position = ModelParameter_Bert["max_position_embeddings"] * ModelParameter_Bert["hidden_size"]

# 分段嵌入（Segment Embeddings）：
# type_vocab_size * hidden_size = 2 * 768
Segment =  ModelParameter_Bert["type_vocab_size"] * ModelParameter_Bert["hidden_size"]

# 2. Transformer编码器层
# 对于每个Transformer层（共num_hidden_layers层），有以下参数：

# 自注意力机制：
# 查询（Queries）、键（Keys）、值（Values）的线性变换矩阵： 3 * hidden_size * hidden_size
# 输出线性变换矩阵：hidden_size * hidden_size

# 前馈网络（FFN）：
# 第一个线性变换矩阵：
# hidden_size * intermediate_size
# 第二个线性变换矩阵：
# intermediate_size * hidden_size

# 层归一化：每层两个参数（缩放和偏移），共
# 2 * hidden_size

# 每个Transformer层的总参数
# = (3 * hidden_size^2 + hidden_size^2) + (hidden_size * intermediate_size + intermediate_size * hidden_size) + 2 * hidden_size
# 所有Transformer层的总参数
# = num_hidden_layers * [(3 * hidden_size^2 + hidden_size^2) + (hidden_size * intermediate_size + intermediate_size * hidden_size) + 2 * hidden_size]

# 将hidden_size和intermediate_size的值代入，得到：
# 每个Transformer层的参数 = (3 * 768^2 + 768^2) + (768 * 3072 + 3072 * 768) + 2 * 768
Transformer_one = ((3 * ModelParameter_Bert["hidden_size"] ** 2 + ModelParameter_Bert["hidden_size"] ** 2) +
                   (ModelParameter_Bert["hidden_size"]  * ModelParameter_Bert["intermediate_size"]  +
                    ModelParameter_Bert["intermediate_size"] * ModelParameter_Bert["hidden_size"]) +
                   2 * ModelParameter_Bert["hidden_size"])

# 所有Transformer层的总参数 = 12 * [(3 * 768^2 + 768^2) + (768 * 3072 + 3072 * 768) + 2 * 768]
Transformer_All = ModelParameter_Bert["num_hidden_layers"] * Transformer_one

# 3. 池化层（对于标准BERT，这主要是[CLS]标记的变换）
# 输入到输出的权重矩阵：input_size * output_size
# 偏置项bias：output_size

# 每个FC层的参数 = 768 * 768 + 768
Pooling_one = ModelParameter_Bert["pooler_fc_size"] * ModelParameter_Bert["pooler_fc_size"] + ModelParameter_Bert["pooler_fc_size"]

# 所有FC层的总参数 = pooler_num_fc_layers * (768 * 768 + 768) = 3 * (768 * 768 + 768)
Pooling_All = ModelParameter_Bert["pooler_num_fc_layers"]  * Pooling_one

# 4. 分类层（如果模型用于分类任务）
# 分类线性层：hidden_size * num_labels = 768 * 18（注意：这里假设有一个具有18个类别的分类任务）

# 可训练参数总数 = 嵌入层参数 + 所有Transformer层参数 + 池化层参数 + 分类层参数
# 嵌入层参数 = 21128 * 768 + 512 * 768 + 2 * 768
Embeddings = Token + Position + Segment

# 分类层参数 = 768 * 18
Classify =  ModelParameter_Bert["hidden_size"] *  ModelParameter_Bert["num_labels"]

# 所有Transformer层参数：
FinalResult = Embeddings + Transformer_All + Pooling_All + Classify

print(FinalResult) # 所有Transformer层可训练参数 = 103359744

