# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    # "train_data_path": "../data/train_tag_news.json",
    # "valid_data_path": "../data/valid_tag_news.json",
    "train_data_path": "./homework_train_data.csv",
    "valid_data_path": "./homework_valid_data.jcsv",
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
    "pretrain_model_path":r"D:\ai预习课件\week6 语言模型和预训练\bert-base-chinese",
    "seed": 987
}

