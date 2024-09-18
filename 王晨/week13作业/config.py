# -*- coding: utf-8 -*-

"""
配置参数信息
"""

class Config:
    def __init__(self):
        self.model_path = "./models/"
        self.bert_path = r"E:\AI\课程资料\bert-base-chinese"
        self.train_data_path = "./data/train.txt"
        self.valid_data_path = "./data/valid.txt"
        self.schema_path = "./data/schema.json"
        self.max_length = 128
        self.batch_size = 32
        self.epoch = 3
        self.class_num = 9
        self.learning_rate = 5e-5
        self.optimizer = "adam"
        self.use_crf = False
        self.tuning_tactics = "lora_tuning"  # LoRA 训练模式
        self.lora_r = 4  # LoRA 参数
        self.lora_alpha = 16
        self.lora_dropout = 0.1


