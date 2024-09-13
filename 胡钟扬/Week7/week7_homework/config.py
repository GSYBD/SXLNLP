# -*- coding: utf-8 -*-

'''
配置参数信息
'''
Config = {
    "model_path": "week7_homework/output",
    "train_data_path": "week7_homework/data/train_data.json",
    "valid_data_path": "week7_homework/data/valid_data.json",
    "vocab_path":"week7_homework/chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 5,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"D:\pre-trained-models\bert-base-chinese",
    "seed": 987,
    "model_list": ['fast_text', 'lstm', 'gru', \
                   'rnn', 'cnn', 'gated_cnn', 'stack_gated_cnn', \
                   'rcnn', 'bert', 'bert_lstm', 'bert_cnn', \
                   'bert_mid_layer']
}