# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train.txt",
    "valid_data_path": "ner_data/test.txt",
    "vocab_path":"/Users/chenmo/workspace-py/demo1/bert-base-chinese/vocab.txt",
    "max_length": 100,
    "hidden_size": 768,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": 9,
    "tuning_tactics": "lora_tuning",
    "bert_path": '/Users/chenmo/workspace-py/demo1/bert-base-chinese'
}

