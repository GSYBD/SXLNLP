# -*- coding: utf-8 -*-

"""
配置参数信息
"""

# Config = {
#     "model_path": "output",
#     "train_data_path": "../data/train_tag_news.json",
#     "valid_data_path": "../data/valid_tag_news.json",
#     "vocab_path":"chars.txt",
#     "model_type":"bert",
#     "max_length": 30,
#     "hidden_size": 256,
#     "kernel_size": 3,
#     "num_layers": 2,
#     "epoch": 15,
#     "batch_size": 128,
#     "pooling_style":"max",
#     "optimizer": "adam",
#     "learning_rate": 1e-3,
#     "pretrain_model_path":r"F:\Desktop\work_space\pretrain_models\bert-base-chinese",
#     "seed": 987
# }


Config = {
    "model_path": "output",
    "train_data_path": "D:/ai/week7 文本分类问题/文本分类练习.csv",
    "valid_data_path": "D:/ai/week7 文本分类问题/文本分类练习_测试正确率.csv",
    "vocab_path":"chars.txt",
    "model_type":"rnn",
    # "model_type":"lstm",
    "max_length": 500,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-1,
    "pretrain_model_path":r"D:\aiproject\bert-base-chinese",
    "seed": 987
}
