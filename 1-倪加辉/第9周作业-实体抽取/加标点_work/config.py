"""
配置参数信息
"""
Config = {
    "model_path": "./output/",
    "model_name": "model.pt",
    "schema_path": r"D:\NLP\video\第九周\week9 序列标注问题\加标点\data\schema.json",
    "train_data_path": r"D:\NLP\video\第九周\week9 序列标注问题\加标点\data\train_corpus",
    "valid_data_path": r"D:\NLP\video\第九周\week9 序列标注问题\加标点\data\valid_corpus",
    "vocab_path": r"D:\NLP\video\第七周\data\vocab.txt",
    "model_type": "bert",
    # 数据标注中计算loss
    "use_crf": False,
    # 文本向量大小
    "char_dim": 20,
    # 文本长度
    "max_len": 50,
    # 词向量大小
    "hidden_size": 64,
    # 训练 轮数
    "epoch_size": 15,
    # 批量大小
    "batch_size": 25,
    # 训练集大小
    "simple_size": 300,
    # 学习率
    "lr": 1e-4,
    # dropout
    "dropout": 0.5,
    # 优化器
    "optimizer": "adam",
    # 卷积核
    "kernel_size": 3,
    # 最大池 or 平均池
    "pooling_style": "max",
    # 模型层数
    "num_layers": 3,
    "bert_model_path": r"D:\NLP\video\第六周\bert-base-chinese",
    # 输出层大小
    "output_size": 4,
    # 随机数种子
    "seed": 987
}

