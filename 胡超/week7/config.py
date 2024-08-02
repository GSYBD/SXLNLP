Config = {
    "model_path": "output",
    "train_data_path": "./data/train_data.json",
    "valid_data_path": "./data/test_data.json",
    "vocab_path": "chars.txt",
    "model_type": "bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 3,
    "batch_size": 128,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "pretrain_model_path": "./bert-base-chinese",
    "seed": 987,
    "save_model": False,
    "label_to_index": {
        "差评": 0,
        "好评": 1
    }
}