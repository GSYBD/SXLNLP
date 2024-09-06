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
    "max_length": 128,
    "hidden_size": 768,
    "num_layers": 12,
    "epoch": 100,
    "batch_size": 10,
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "vocab_size": 21128,
    "bert_path": r"D:\code\github\bert-base-chinese",
    "corpus_path": r"D:\code\data\week11_data\sample_data.json",
    'train_sample': 2000
}

