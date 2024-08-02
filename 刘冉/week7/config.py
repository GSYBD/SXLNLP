# -*- coding: utf-8 -*-
from enum import Enum
'''
配置参数
'''

#fast_text cnn rnn lstm gru gated_cnn stack_gated_cnn rcnn bert bert_lstm bert_cnn bert_mid_layer
Config = {
    "model_path": "output",
    "train_data_path": "data/train.json",
    "valid_data_path": "data/valid.json",
    "predict_data_path": "data/predict.json",
    "vocab_path": "chars.txt",
    "model_type": "cnn",
    "max_length": 463,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 2,
    "batch_size": 10,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": r"/Users/liuran/bert-base-chinese ",
    "seed": 996
}