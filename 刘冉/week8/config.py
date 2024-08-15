# coding: utf-8

"""
参数配置
"""

Config = {

    "model_path": "model_output",
    "schema_path": "data/schema.json",
    "train_data_path": "data/train.json",
    "valid_data_path": "data/valid.json",
    "vocab_path": "data/chars.txt",
    "max_length": 20,
    "hidden_size": 128,
    "kernel_size": 3,
    "epoch": 50,
    "batch_size": 20,
    "epoch_data_size": 10000,  # 每轮训练中采样数量
    "positive_sample_rate": 0.5,  # 正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "margin": 1.0,

}