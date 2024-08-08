# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "data/train_set.json",
    "valid_data_path": "data/valid_set.json",
    "vocab_path":"chars.txt",
    "model_type":"bert_lstm",
    "max_length": 470, # 最大长度是463，留些富裕
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 32,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"F:\课程\八斗精品班\第六周 预训练模型\bert-base-chinese",
    "seed": 987
}

