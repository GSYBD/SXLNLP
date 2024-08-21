# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":r"D:\tys\note\1.ai\资料库\第二周\week2 深度学习基本原理\week6 语言模型和预训练\bert-base-chinese\vocab.txt",
    "max_length": 100,
    "hidden_size": 384,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": 9,
    "pretrain_model_path":r"D:\tys\note\1.ai\资料库\第二周\week2 深度学习基本原理\week6 语言模型和预训练\bert-base-chinese",
}

