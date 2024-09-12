# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": r"D:\NLP\video\第九周 序列标注任务\课件\week9 序列标注问题\ner\ner_data\schema.json",
    "train_data_path": r"D:\NLP\video\第九周 序列标注任务\课件\week9 序列标注问题\ner\ner_data\train.txt",
    "valid_data_path": r"D:\NLP\video\第九周 序列标注任务\课件\week9 序列标注问题\ner\ner_data\test.txt",
    "vocab_path": r"D:\NLP\video\第九周 序列标注任务\课件\week9 序列标注问题\ner\chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 8,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"D:\NLP\video\第六周\bert-base-chinese",
    "tuning_tactics": "lora_tuning"
}