# -*- coding: utf-8 -*-

"""
配置参数信息
"""

# Config = {
#     "model_path": "/home/quintus/badou/week9/model_output",
#     "schema_path": "/home/quintus/badou/week9/ner_data/schema.json",
#     "train_data_path": "/home/quintus/badou/week9/ner_data/train",
#     "valid_data_path": "/home/quintus/badou/week9/ner_data/test",
#     "vocab_path":"/home/quintus/badou/week9/chars.txt",
#     "max_length": 100,
#     "hidden_size": 256,
#     "num_layers": 2,
#     "epoch": 20,
#     "batch_size": 16,
#     "optimizer": "adam",
#     "learning_rate": 1e-3,
#     "use_crf": False,
#     "class_num": 9,
#     "bert_path": r"/media/quintus/新加卷/zyj/八斗精品班/第六周 预训练模型/bert-base-chinese"
# }

Config = {
    "model_path": "/home/quintus/badou/week9/model_output",
    "schema_path": "/home/quintus/badou/week9/ner_data/schema.json",
    "train_data_path": "/home/quintus/badou/week9/ner_data/train",
    "valid_data_path": "/home/quintus/badou/week9/ner_data/test",
    "vocab_path":"/home/quintus/badou/week9/chars.txt",
    "max_length": 100,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-5,  # Lower learning rate for BERT fine-tuning
    "use_crf": False,  # Set to True if you want to use a CRF layer
    "class_num": 9,  # Number of NER tags
    "bert_model_name": "bert-base-chinese",  # Update this to the correct BERT model name
    "bert_path": r"/media/quintus/新加卷/zyj/八斗精品班/第六周 预训练模型/bert-base-chinese"
}
