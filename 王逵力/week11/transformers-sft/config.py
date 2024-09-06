Config = {
    "pretrained_model_name": "bert-base-uncased",  # 预训练模型名称，可以更换为其他模型
    "num_labels": 2,  # 标签类别数，依据你的任务调整
    "learning_rate": 1e-5,  # 微调时的学习率
    "epoch": 3,  # 训练的轮次
    "max_length": 128,  # 输入序列的最大长度
    "train_data_path": "sample_data.json",  # 训练数据文件路径
    "model_path": "./model/",  # 保存模型的路径
    "optimizer": "adam",  # 优化器类型
}

