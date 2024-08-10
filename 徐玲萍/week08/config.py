"""
配置参数信息
"""
Config = {
    "model_path": "./output/",
    "model_name": "model.pt",
    "schema_path": r"/Users/xulingping/Documents/nlp_codes/week8_nlp_xlp/data/schema.json",
    "train_data_path": r"/Users/xulingping/Documents/nlp_codes/week8_nlp_xlp/data/data.json",
    "valid_data_path": r"/Users/xulingping/Documents/nlp_codes/week8_nlp_xlp/data/valid.json",
    "vocab_path": r"/Users/xulingping/Documents/nlp_codes/week7_nlp_xlp/nn_pipline/bert-base-chinese/vocab.txt",
    "model_type": "rnn",
    # 正样本比例
    "positive_sample_rate": 0.5,
    "use_bert": False,
    # 文本向量大小
    "char_dim": 32,
    # 文本长度
    "max_len": 20,
    # 词向量大小
    "hidden_size": 128,
    # 训练 轮数
    "epoch_size": 15,
    # 批量大小
    "batch_size": 32,
    # 训练集大小
    "simple_size": 300,
    # 学习率
    "lr": 1e-3,
    # dropout
    "dropout": 0.5,
    # 优化器
    "optimizer": "adam",
    # 卷积核
    "kernel_size": 3,
    # 最大池 or 平均池
    "pooling_style": "max",
    # 模型层数
    "num_layers": 2,
    #"bert_model_path": r"D:\NLP\video\第六周\bert-base-chinese",
    # 输出层大小
    "output_size": 2,
    # 随机数种子
    "seed": 987
}
