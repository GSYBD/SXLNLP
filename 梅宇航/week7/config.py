# config.py
class Config:
    model_path = "output"
    train_data_path = "D:/BaiduNetdiskDownload/week7 文本分类问题/文本分类练习数据集/文本分类练习.csv"
    valid_data_path = "D:/BaiduNetdiskDownload/week7 文本分类问题/文本分类练习数据集/文本分类练习.csv"
    vocab_path = "D:/BaiduNetdiskDownload/week7 文本分类问题/文本分类练习数据集/vocab.json"
    model_type = "cnn"
    max_length = 30
    hidden_size = 128
    kernel_size = 3
    num_layers = 2
    epoch = 10
    batch_size = 32
    pooling_style = "max"
    optimizer = "adam"
    learning_rate = 1e-3
    pretrain_model_path = "bert-base-chinese"
    seed = 42
    class_num = 2
    vocab_size = 5000
