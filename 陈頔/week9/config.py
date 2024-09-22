# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "./ner_data/schema.json",
    "train_data_path": "./ner_data/train",
    "valid_data_path": "./ner_data/test",
    # 修改bert的路径
    "vocab_path":r"D:\ai预习课件\week6 语言模型和预训练\bert-base-chinese\vocab.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": 9,
    # bert的路径
    "bert_path": r"D:\ai预习课件\week6 语言模型和预训练\bert-base-chinese"
}

