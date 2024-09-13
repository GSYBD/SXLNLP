# -*- coding: utf-8 -*-
import os.path as osp

ASSIGNMENT_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": osp.join(ASSIGNMENT_DIR, "data/schema.json"),
    "train_data_path":  osp.join(ASSIGNMENT_DIR, "data/train.json"),
    "valid_data_path":  osp.join(ASSIGNMENT_DIR, "data/valid.json"),
    "vocab_path": osp.join(ASSIGNMENT_DIR, "chars.txt"),
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 32,
    "epoch_data_size": 200,     #每轮训练中采样数量
    "positive_sample_rate":0.5,  #正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-3,
}