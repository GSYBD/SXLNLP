# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 3,
    "epoch": 20,
    "batch_size": 16,
    "tuning_tactics":"lora_tuning",
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"C:\Users\22807\Desktop\LearnPython\我的练习和作业\bert-base-chinese",
    "seed": 987
}

