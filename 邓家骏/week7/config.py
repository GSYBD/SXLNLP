# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": r"邓家骏\week7\model",
    "train_data_path": r"D:\code\data\week7_data\train.csv",
    "valid_data_path": r"D:\code\data\week7_data\vaild.csv",
    "vocab_path":r"邓家骏\week7\chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 100,
    "batch_size": 128,
    "pooling_style":"avg",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"D:\code\github\bert-base-chinese",
    "seed": 987,
}

