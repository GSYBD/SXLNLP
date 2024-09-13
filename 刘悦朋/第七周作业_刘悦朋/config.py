"""

    配置参数信息

"""

Config = {
    # loader
    'vocab_path': 'chars.txt',
    'max_length': 30,
    'batch_size': 32,
    'pretrain_model_path': 'bert-base-chinese',

    # model
    'hidden_size': 768,
    'model_type': 'bert_lstm',
    'num_layers': 1,
    'pooling_style': 'avg',
    'optimizer': 'adam',
    'learning_rate': 1e-3,

    # main
    'seed': 42,
    'model_path': 'output',
    'train_data_path': '文本分类练习.csv',
    'epoch': 20,

    # evaluate
    'valid_data_path': '文本分类练习.csv',
}
