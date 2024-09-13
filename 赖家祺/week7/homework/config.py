# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "./train_valid_data/train_data.txt",
    "valid_data_path": "./train_valid_data/valid_data.txt",
    "vocab_path": "chars.txt",
    "model_type": "bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 128,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    # "pretrain_model_path":r"F:\Desktop\work_space\pretrain_models\bert-base-chinese",
    "pretrain_model_path": r"D:\八斗人工智能\nlp\0-bert-base-chinese\bert-base-chinese",
    "seed": 987
}
