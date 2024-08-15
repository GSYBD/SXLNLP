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
        anchor = anchor.float().cuda()
        positive = positive.float().cuda()
        negative = negative.float().cuda()

    # 加载优化器
    optimizer = choose_optimizer(config, model)

    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)

    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []

        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            anchor, positive, negative, labels = batch_data  # 获取三元组

            # 确保 anchor, positive, negative 都是 float 类型并设置 requires_grad=True
            anchor = anchor.float().requires_grad_()
            positive = positive.float().requires_grad_()
            negative = negative.float().requires_grad_()

            # 计算损失
            loss = model.cosine_triplet_loss(anchor, positive, negative)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)

    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return


if __name__ == "__main__":
    main(Config)