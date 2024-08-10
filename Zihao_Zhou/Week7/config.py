# -*- coding: utf-8 -*-

"""
配置参数信息
"""
#"bert, lstm, gru, rnn, cnn, gated_cnn, stack_gated_cnn, rcnn"

Config = {
    "model_path": "output",
    "train_data_path": "../train.txt",
    "valid_data_path": "../test.txt",
    "vocab_path": "chars.txt",
    "model_type": "rcnn",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-4,
    "pretrain_model_path": r"E:\NLP学习\第六周 预训练模型\bert-base-chinese",
    "seed": 2024,
    "class_num": 2
}

