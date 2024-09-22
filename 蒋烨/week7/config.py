# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "data/train.json",
    "valid_data_path": "data/valid.json",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 80,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 1,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":"bert-base-chinese",
    "seed": 987
}

