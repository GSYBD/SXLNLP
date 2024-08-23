# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": r"邓家骏\week9\ner\model_output",
    "schema_path": r"D:\code\data\week9_data\ner_data\schema.json",
    "train_data_path": r"D:\code\data\week9_data\ner_data\train",
    "valid_data_path": r"D:\code\data\week9_data\ner_data\test",
    "vocab_path":"chars.txt",
    "max_length": 100,
    "hidden_size": 768,
    "num_layers": 12,
    "epoch": 25,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-6,
    "use_crf": True,
    "class_num": 9,
    "bert_path": r"D:\code\github\bert-base-chinese"
}

