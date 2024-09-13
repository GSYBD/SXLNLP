# coding: utf-8
import os.path

import torch
import logging
import numpy as np
from loader import load_data
from model import PunctuationModel, choose_optimizer
from evaluate import Evaluator
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

'''
模型训练

'''


def main(config):
    # 导出模型
    mode_path = config["model_path"]
    if not os.path.isdir(mode_path):
        os.mkdir(mode_path)
    # 加载训练数据
    train_data = load_data(config, config["train_path"])
    # model
    model = PunctuationModel(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 效果测试
    evalutor = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [b.cuda() for b in batch_data]
            sentences, labels = batch_data
            loss = model(sentences, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        logger.info("epoch %d 平均 loss: %f" % (epoch, np.mean(train_loss)))
        evalutor.eval(epoch)
        if epoch % 10 == 0:
            out_model_path = os.path.join(mode_path, "punctuation_model_"+str(epoch)+".pth")
            torch.save(model.state_dict(), out_model_path)
    out_model_path = os.path.join(mode_path, "punctuation_model.pth")
    torch.save(model.state_dict(), out_model_path)
    return model


if __name__ == "__main__":
    main(Config)
