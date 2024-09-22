# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": r"D:\AI课程\八斗AI\第十三周 大语言模型微调\homework\ner_data\schema.json",
    "train_data_path": r"D:\AI课程\八斗AI\第十三周 大语言模型微调\homework\ner_data\train.txt",
    "valid_data_path": r"D:\AI课程\八斗AI\第十三周 大语言模型微调\homework\ner_data\test.txt",
    "vocab_path": r"D:\AI课程\八斗AI\第十三周 大语言模型微调\homework\chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 8,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"D:\AI课程\八斗AI\第六周 预训练模型\bert-base-chinese",
    "tuning_tactics": "lora_tuning"
}