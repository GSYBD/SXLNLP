# -*- coding: utf-8 -*-
import torch
import os
import numpy as np
import logging
from config import Config
from model import BertCRF, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    train_data = load_data(config["train_data_path"], config)
    model = BertCRF(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)

    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for batch_data in train_data:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch_data
            if cuda_flag:
                input_ids, attention_mask, labels = [d.cuda() for d in [input_ids, attention_mask, labels]]
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if len(train_loss) % 10 == 0:
                logger.info("batch loss %f" % np.mean(train_loss))
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)

    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return model, train_data

if __name__ == "__main__":
    model, train_data = main(Config)
