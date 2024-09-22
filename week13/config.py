# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path": "chars.txt",
    "model_type": "bert",
    "max_length": 100,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 8,
    "epoch": 10,
    "batch_size": 64,
    "tuning_tactics": "lora_tuning",
    # "tuning_tactics":"finetuing",
    "pooling_style": "max",
    "optimizer": "adam",
    "use_crf": False,
    "learning_rate": 1e-3,
    "class_num": 9,
    "pretrain_model_path": r"F:\学习资料\PYTHON\课程\bert-base-chinese",
    "seed": 987
}
