# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "outpsut",
    "train_data_path": "D:\BaiduNetdiskDownload\第七周 文本分类问题\week7 文本分类问题\data_train.csv",
    "valid_data_path": "D:\BaiduNetdiskDownload\第七周 文本分类问题\week7 文本分类问题\date_valid.csv",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"D:\BaiduNetdiskDownload\第六周 预训练模型\bert-base-chinese",
    "seed": 987
}

