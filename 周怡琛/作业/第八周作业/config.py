
# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "./output/",
    "model_name": "model.pt",
    "schema_path": r"D:\badou_nlp_program\week8\data\schema.json",
    "train_data_path": r"D:\badou_nlp_program\week8\data\data.json",
    "valid_data_path": r"D:\badou_nlp_program\week8\data\valid.json",
    "vocab_path": r"D:\badou_nlp_program\week7\chars.txt",
    "model_type": "rnn",
    "positive_sample_rate": 0.5,    # 正样本比例
    "use_bert": False,
    "char_dim": 32,                 # 文本向量大小
    "max_len": 20,                  # 文本长度
    "hidden_size": 128,             # 词向量大小
    "epoch_size": 15,               # 训练 轮数
    "batch_size": 32,               # 批量大小
    "simple_size": 300,             # 训练集大小
    "lr": 1e-3,                     # 学习率
    "dropout": 0.5,
    "optimizer": "adam",            # 优化器
    "kernel_size": 3,               # 卷积核
    "pooling_style": "max",
    "num_layers": 2,                # 模型层数
    "bert_model_path": r"D:\bert-base-chinese",
    "output_size": 2,               # 输出层大小
    "seed": 8                       # 随机数种子
}
