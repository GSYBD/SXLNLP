"""
配置参数
"""

config = {
    "model_path": "model",
    "train_data_path": "data/文本分类练习.csv",
    "valid_data_path": "data/文本分类练习.csv",
    "vocab_path": "data/chars.txt",
    "hidden_size": 256,
    "model_type": "bert",
    "num_layers": 2,
    "class_num": 2,
    "pretrain_model_path": r"D:\EVPlayer\八斗精品班\bert-base-chinese",
    "pooling_style": "max",
    "epoch": 20,
    "batch_size": 256,
    "lr": 1e-3,
    "optim": "Adam",
    "max_length": 30
}
