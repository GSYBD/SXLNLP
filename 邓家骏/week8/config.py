# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "邓家骏\week8\model_output",
    "schema_path": r"D:\code\data\week8_data\schema.json",
    "train_data_path": r"D:\code\data\week8_data\train.json",
    "valid_data_path": r"D:\code\data\week8_data\valid.json",
    "vocab_path":"D:\code\data\week8_data\chars.txt",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 32,
    "epoch_data_size": 200,     #每轮训练中采样数量
    "positive_sample_rate":0.5,  #正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "train_with_target": True
}