# 项目实现
## 1. 描述
基于特定的Bert配置文件计算可训练的参数
## 2. 具体实现
`详细计算方法以及说明已经在code里，这里主要叙述一下框架`
1. Embedding层可训练参数的计算
2. Transformer层可训练参数的计算
3. Pooler层可训练参数的计算
4. 根据当前模型暂未计算MLM Head和Classification Head的可训练参数
## 3. 备注
基于下方的Bert配置信息，Bert可训练的参数为24301056
```json
{
  "architectures": [
    "BertForMaskedLM"
  ],
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
  "num_hidden_layers": 1,
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
```