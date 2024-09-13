# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "F:\\nlp\\week7\\train.json",
    "valid_data_path": "F:\\nlp\\week7\\valid.json",
    "vocab_path":"F:\\nlp\\week7\\nn_pipline\\chars.txt",
    "model_type":"fast_text",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"models/bert",
    "seed": 987
}

