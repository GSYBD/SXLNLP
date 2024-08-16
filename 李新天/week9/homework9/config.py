# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "data/schema.json",
    "train_data_path": "data/train_corpus",
    "valid_data_path": "data/valid_corpus",
    "vocab_path":"chars.txt",
    "max_length": 50,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 128,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": None
}