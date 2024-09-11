# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "D:/资料/week7 文本分类问题/week7 文本分类问题/train_data.csv",
    "valid_data_path": "D:/资料/week7 文本分类问题/week7 文本分类问题/test_data.csv",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 3,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"D:\资料\week6 语言模型和预训练\下午\bert-base-chinese",
    "seed": 987
}

