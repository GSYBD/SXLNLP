# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "data/data_train.csv",
    "valid_data_path": "data/date_valid.csv",
    "vocab_path": "bert-base-chinese/vocab.txt",
    "model_type": "bert",
    "max_length": 50,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 1,
    "batch_size": 128,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": r"bert-base-chinese",
    "seed": 987
}
