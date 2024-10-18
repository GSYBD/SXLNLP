# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "train_data_path": "../triplet_data/train_triplet_data.json",
    "valid_data_path": "../triplet_data/valid_triplet_data.json",
    "vocab_path":"chars.txt",
    "schema_path":"schema.json",
    "max_length": 200,
    "hidden_size": 256,
    "epoch": 15,
    "batch_size": 128,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "attribute_loss_ratio":0.01
}