# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "../data_test/train_tag_news.json",
    "valid_data_path": "../data_test/valid_tag_news.json",
    "predict_data_path": "../data_test/predict_tag_news.json",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 768,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"C:\Users\86358\Desktop\nlp\课件\week6 语言模型\bert-base-chinese",
    "seed": 987
}

