# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path": "chars.txt",
    "max_length": 100,
    # "hidden_size": 256,
    "hidden_size": 768,
    "num_layers": 1,
    "epoch": 1,
    "batch_size": 20,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "class_num": 9,
    "bert_path": r"D:\aiproject\bert-base-chinese",
    "peft_type": "lora", #promit_tuning,preft_tuning,p_tuning,adapter,lora

    "model_type": "bert"
}
