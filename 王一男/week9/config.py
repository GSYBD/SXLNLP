# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "data/schema.json",
    "train_data_path": "data/train_corpus",
    "valid_data_path": "data/valid_corpus",
    "vocab_path": "chars.txt",
    "max_length": 50,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 128,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": True,
    "class_num": 4,
    "pretrain_model_path": r"E:\111绝密资料，禁止外传(2)\AIML_llm\第六周 预训练模型\bert-base-chinese",
    "use_bert": True
}

# bert怎么用词表/load时候encoder转化为label首位增加cls占位符
# 模型训练占用资源和模型大小关系并没有那么大。与max_size等影响张量长度有关。还有batch_size。

