# -*- coding: utf-8 -*-
import sys
import torch
import os
import random
import os
import numpy as np
import time
import logging
import json
from config import Config
from evaluate import Evaluator
from loader import load_data
from model import LanguageModel


#这个transformer是本文件夹下的代码，和我们之前用来调用bert的transformers第三方库是两回事
from transformer.Models import Transformer

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


"""
模型训练主程序
"""

# seed = Config["seed"]
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)


def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载模型
    logger.info(json.dumps(config, ensure_ascii=False, indent=2))
    model = LanguageModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        if cuda_flag:
            model.cuda()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_seq, output_seq = batch_data
            optimizer.zero_grad()
            loss=model(input_seq,output_seq)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        logger.info("epoch average loss: %f" % np.mean(train_loss))
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return

if __name__ == "__main__":

    main(Config)


