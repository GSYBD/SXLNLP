# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "文本分类练习.csv",
    "valid_data_path": "文本分类练习.csv",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 1,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "pretrain_model_path":r"D:\AI课程\八斗AI\第六周 预训练模型\bert-base-chinese",
    "seed": 987
}

