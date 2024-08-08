import torch
import random
import numpy as np

# 配置和超参数设置
Config = {
    "seed": 42,
    "max_len": 50,
    "test_size": 0.2,
    "batch_size": 20,
    "epoch_num": 20,
    "char_dim": 768,
    "lr": 1e-5
}

# 设置随机种子
seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
