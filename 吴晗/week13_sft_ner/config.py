# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": r"吴晗\week9_bert_ner\ner_data\schema.json",
    "train_data_path": r"吴晗\week9_bert_ner\ner_data\train.txt",
    "valid_data_path": r"吴晗\week9_bert_ner\ner_data\test.txt",
    "vocab_path": r"chars.txt",
    "max_length": 150,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"F:\课程\八斗精品班\第六周 预训练模型\bert-base-chinese",
    "tuning_tactics": "lora_tuning",
    "seed": 42
}