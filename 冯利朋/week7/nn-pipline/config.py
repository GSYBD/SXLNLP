"""
配置信息,用于统一配置
输入模型配置参数，如学习率等等
"""
Config = {
    "model_path": "output",  # 模型的保存路径
    "train_data_path": "./data/train_data.csv",  # 训练数据集路径
    "valid_data_path": "./data/valid_data.csv",  # 验证数据集路径
    "test_data_path": "./data/test_data.csv",  # 测试数据集路径
    "vocab_path": "./chars.txt",  # 字表路径
    "model_type": "bert",  # 模型的训练模式
    "use_bert": True,   # 是否使用bert 要和model_type对应
    "class_num": 2,
    "max_length": 30,  # 文本最大长度
    "hidden_size": 256,  # embedding层hidden_size
    "kernel_size": 3,  # 核大小,如果使用CNN
    "num_layers": 2,  # RNN层数
    "epoch": 10,  # 训练次数
    "batch_size": 64,  # 每次训练的样本数量 报显存错误的时候可以调整这个，或者是bert的层数
    "pooling_style": "avg",  # 池化层类型
    "optimizer": "adam",  # 优化器类型
    "learning_rate": 1e-5,  # 学习率 如果使用bert则这个学习率不需要太大，因为bert已经学习的差不多了 1e-5
    "pretrain_model_path": "/Users/gonghengan/Documents/hugging-face/bert-base-chinese",  # bert路径
    "seed": 987
}

if __name__ == '__main__':
    path = "./chars.txt"
