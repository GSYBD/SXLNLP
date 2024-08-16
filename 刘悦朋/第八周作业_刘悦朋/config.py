"""

    配置参数信息

"""

Config = {
    # loader.py
    'schema_path': 'data/schema.json',
    'vocab_path': 'chars.txt',
    'max_length': 20,
    'epoch_data_size': 200,  # 每轮训练中采样数量
    'positive_sample_rate': 0.5,  # 正样本比例
    'batch_size': 32,

    # model.py
    'hidden_size': 128,

    # main.py
    'model_path': 'model_output',
    'train_data_path': 'data/train.json',
    'optimizer': 'adam',
    'learning_rate': 1e-3,
    'epoch': 20,

    # evaluate.py
    'valid_data_path': 'data/valid.json',
}
