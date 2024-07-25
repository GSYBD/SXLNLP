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


计算可训练参数数量
