# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": r"C:\Users\Mrhuang\Desktop\myNLP\SXLNLP\黄雨石\week7\nn_pipline\data\train_set.csv",
    "valid_data_path": r"C:\Users\Mrhuang\Desktop\myNLP\SXLNLP\黄雨石\week7\nn_pipline\data\validation_set.csv",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 1,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"C:\Users\Mrhuang\Desktop\myNLP\SXLNLP\黄雨石\week7\bert-base-chinese",
    "seed": 987
}

