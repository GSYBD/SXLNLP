# -*- coding: utf-8 -*-

import os
"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "D:/project/git/badouai/week7/data/train_str_class.csv",
    "valid_data_path": "D:/project/git/badouai/week7/data/valid_str_class.csv",
    # "train_data_path": "D:/project/git/badouai/week7/data/train_tag_news.json",
    # "valid_data_path": "D:/project/git/badouai/week7/data/valid_tag_news.json",
    "vocab_path": r"D:/project/git/badouai/week7/nn_pipline/chars.txt",
    "model_type": "bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": r"D:/project/git/badouai/week6/bert-base-chinese",
    "seed": 987
}

