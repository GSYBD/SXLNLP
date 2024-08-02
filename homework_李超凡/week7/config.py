# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": r"F:\NLP\Code\Project\SXLNLP\homework_李超凡\week7\output",
    "vocab_path":"chars.txt",
    "vocab_size":4622,
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":"F:/NLP/pretrain_models/bert-base-chinese",
    "train_data_path":r"F:\NLP\Code\Project\SXLNLP\homework_李超凡\week7\dataset\train_data.csv",
    "valid_data_path":r"F:\NLP\Code\Project\SXLNLP\homework_李超凡\week7\dataset\valid_data.csv",
    "seed": 987,
    "train_size":0.8,
    "class_num":2
}

