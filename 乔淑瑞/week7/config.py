# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "data/train_data.txt",
    "valid_data_path": "data/valid_data.txt",
    "vocab_path": "chars.txt",
    "model_type": "bert",
    "max_length": 30,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 3,
    "batch_size": 128,
    "pooling_style": "max",
    "optimizer": "adam",
    "test_size": 0.2,
    "learning_rate": 1e-3,
    "pretrain_model_path": r"D:\material\八斗\第六周 预训练模型\bert-base-chinese",
    "seed": 987
}
