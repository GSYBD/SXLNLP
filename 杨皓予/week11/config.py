# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": r"D:\资料\week10 文本生成问题\bert_sft\sample_data.json",
    "valid_data_path": "ner_data/test",
    "vocab_path": r"D:\资料\week9 序列标注问题\week9 序列标注问题\ner\bert-base-chinese\vocab.txt",
    "max_length": 150,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 20,
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"D:\资料\week9 序列标注问题\week9 序列标注问题\ner\bert-base-chinese"
}

