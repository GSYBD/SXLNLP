"""
配置类
"""
Config = {
    "model_path": "model_output",
    "schema_path": "./ner_data/schema.json",
    "train_data_path": './ner_data/train.txt',
    "valid_data_path": './ner_data/test.txt',
    "vocab_path": "chars.txt",
    "max_length": 150,
    "hidden_size": 256,
    "epoch": 10,
    "batch_size": 64,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": False,
    "use_bert": True,
    "class_num": 9,
    "pretrain_model_path": "D:\appdev\PyProject\Py_AI\第六周预训练模型\bert-base-chinese",  # bert路径
    "tuning_tactics":"lora_tuning",

}