# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    # "train_data_path": "../data/train_tag_news.json",
    # "valid_data_path": "../data/valid_tag_news.json",
    "my_train_data_path": "../data/my_train_data.csv",
    "my_val_data_path": "../data/my_val_data.csv",
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
    # "pretrain_model_path":r"F:\Desktop\work_space\pretrain_models\bert-base-chinese",
    "pretrain_model_path":r"D:\my_study\4_八斗AI\0_八斗精品班\6_第6周_预训练模型\bert-base-chinese",
    "seed": 987
}

