
"""
配置参数信息  mode_type =[Triplet,Twins]
"""

config ={
    'train_path':'../data/train.json',
    'vocab_path':'./chars.txt',
    "valid_path":'../data/valid.json',
    "schema_path":'../data/schema.json',
    "epoch_num":5,
    "batch_size":32,
    'sentence_len': 7,
    "one_zero_rate":0.5,
    "all_train_data_size":200,
    "learning_rate":0.01,
    "hidden_size":128,
    "optimizer":'adam',
    "margin":0.1,
    "mode_type":"Triplet"

}