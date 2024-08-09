# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)

    # 加载模型
    model = SiameseNetwork(config)

    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU can be used, migrating model to GPU")
        model = model.cuda()

    # 加载优化器
    optimizer = choose_optimizer(config, model)

    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)

    # 训练
    for epoch in range(config["epoch"]):
        model.train()
        logger.info(f"Epoch {epoch + 1} begin")
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id1, input_id2, input_id3 = batch_data
            # 前向传播
            vector1, vector2, vector3 = model(input_id1, input_id2, input_id3)
            # 计算损失
            loss = model.cosine_triplet_loss(vector1, vector2, vector3)
            train_loss.append(loss.item())
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = np.mean(train_loss)
        logger.info(f"Epoch average loss: {avg_loss:.6f}")
        evaluator.eval(epoch)

    model_path = os.path.join(config["model_path"], f"model.pth")
    torch.save(model.state_dict(), model_path)

    return


if __name__ == "__main__":
    main(Config)