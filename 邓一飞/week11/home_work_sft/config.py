# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {

    # "model_type": "transformer",
    "model_type": "bert",
    # "model_type": "linear",

    "bert_model_path": "D:/aiproject/bert-base-chinese",

    "model_path_save": "",
    "schema_path": "../data/schema.json",
    "train_data_path": "./data/sample_data.json",
    "valid_data_path": "../data/valid.json",
    "vocab_path": "D:/aiproject/A002/data/chars.txt",
    "max_length": 200, #文本长度
    "positive_sample_rate": 0.5,  # 正样本比例

    "epoch_num": 10,
    "batch_size": 32,
    "epoch_data_size": 320,  # 每轮训练中采样数量
    "hidden_size": 128,
    "embedding_dim": 768,
    "learning_rate": 1e-3,
    "optimizer": "adam",

}
