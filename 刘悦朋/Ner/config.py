"""

    配置参数信息

"""

Config = {
    # loader.py
    'vocab_path': 'chars.txt',
    'schema_path': 'ner_data/schema.json',
    'max_length': 50,
    "batch_size": 32,
    'bert_path': '../../bert-base-chinese',

    # model.py
    'pretrain_model_path': '../../bert-base-chinese',
    'class_num': 9,
    'use_crf': False,
    'optimizer': 'adam',
    'learning_rate': 1e-3,

    # main.py
    'seed': 42,
    "model_path": "model_output",
    'num_gpus': 1,
    "train_data_path": "ner_data/train",
    'epoch': 20,

    'tuning_tactics': 'lora_tuning',

    # evaluate.py
    'valid_data_path': 'ner_data/test',
}
