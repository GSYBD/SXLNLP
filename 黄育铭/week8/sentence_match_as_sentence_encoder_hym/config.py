# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "D:/八斗精品班/第八周 文本匹配/week8 文本匹配问题//data/schema.json",
    "train_data_path": "D:/八斗精品班/第八周 文本匹配/week8 文本匹配问题//data/train.json",
    "valid_data_path": "D:/八斗精品班/第八周 文本匹配/week8 文本匹配问题//data/valid.json",
    "vocab_path":"D:/八斗精品班/第八周 文本匹配/week8 文本匹配问题/chars.txt",
    "max_length": 20,
    "hidden_size": 256,
    "epoch": 20,
    "batch_size": 10,
    "epoch_data_size": 1,     #每轮训练中采样数量
    "positive_sample_rate":0.5,  #正样本比例
    "optimizer": "adam",
    "learning_rate": 0.00001,
    "vocab_size":4000,
}