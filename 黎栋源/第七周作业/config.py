Config = {
    "model_path": "output",
    "data_path": "文本分类练习.csv",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "test_size":0.3,#测试集比例
    "random_state":42,
    "pretrain_model_path":r"E:\NLP\bert-base-chinese",
    "seed": 987
}
