# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "tuning_tactics": "lora_tuning",
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path": r"D:\资料\week13 大语言模型相关第三讲\ner\bert-base-chinese\vocab.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 8,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"D:\资料\week13 大语言模型相关第三讲\ner\bert-base-chinese"

}

