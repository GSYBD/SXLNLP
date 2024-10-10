# -*- coding: utf-8 -*-

"""
modify your model parameter configuration
"""

import os

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 2,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": True,
    "class_num": 9,
    "bert_path": r"D:\pre-trained-models\bert-base-chinese",
    
    "bert_config":{
        "max_length": 128,
        "hidden_size": 768,
        "num_layers": 12,
        "batch_size": 16,
        "learning_rate": 1e-5,
        "bert_model_path": r"D:\pre-trained-models\bert-base-chinese",
        "dropout": 0.1
    },
    "sentence_config":{
        "max_length": 50,
        "epoch": 10,
        "batch_size": 10,
        "optimizer": "adam",
        "learning_rate":1e-5,
        "seed":42,
        "num_labels": 3,
        "recurrent":"gru",
        "max_sentence": 50,
        "train_data_path": "sentence_ner_data/train",
        "valid_data_path": "sentence_ner_data/test",
    }
}

if "CUDA_VISIBLE_DEVICES" in os.environ:
    Config["num_gpus"] = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
else:
    Config["num_gpus"] = 0

