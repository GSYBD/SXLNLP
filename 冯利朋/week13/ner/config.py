"""
配置类
"""
Config = {
    "model_path": "model_output",  # 模型的输出路径
    "schema_path": "./ner_data/schema.json",  # 类别集
    "train_data_path": './ner_data/train.txt',  # 训练集路径
    "valid_data_path": './ner_data/test.txt',  # 测试集路径
    "vocab_path": "chars.txt",  # 字典路径
    "max_length": 150,  # 文本最大长度
    "hidden_size": 256,
    "epoch": 10,  # 训练轮数
    "batch_size": 64,  # 一次训练的数据量
    "optimizer": "adam",  # 优化器类型
    "learning_rate": 1e-3,  # 学习率
    "use_crf": False,   # 是否使用crt
    "use_bert": True,  # 是否使用bert
    "class_num": 9,     # 标签集数量
    "pretrain_model_path": "/Users/gonghengan/Documents/hugging-face/bert-base-chinese",  # bert路径
    "tuning_tactics":"lora_tuning",

}