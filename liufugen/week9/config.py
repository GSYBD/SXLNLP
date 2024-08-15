# -*- coding: utf-8 -*-

"""
配置参数信息
"""

# Config = {
#     "model_path": "output",
#     "schema_path": "data/schema.json",
#     "train_data_path": "data/train",
#     "valid_data_path": "data/test",
#     "vocab_path":"chars.txt",
#     "max_length": 100,
#     "hidden_size": 256,
#     "num_layers": 2,
#     "epoch": 20,
#     "batch_size": 16,
#     "optimizer": "adam",
#     "learning_rate": 1e-3,
#     "use_crf": False,
#     "class_num": 9,
#     "bert_path": r"../bert-base-chinese"
# }

Config = {
    "model_path": "output",
    "schema_path": "data/schema.json",
    "train_data_path": "data/train",
    "valid_data_path": "data/test",
    "vocab_path":"../bert-base-chinese/vocab.txt",
    "max_length": 150,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 5,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": True,
    "class_num": 9,
    "bert_path": r"../bert-base-chinese"
}


