# coding: utf-8

import torch
import os
import numpy as np
import logging
from config import Config
from model import TripletLossModel, choose_optimizer
from loader import load_data
from evaluate import Evaluator
from predict import Predictor
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

'''
模型训练主程序
'''

def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载模型训练数据
    train_data = load_data(config["train_data_path"],config)
    # 加载模型
    model = TripletLossModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, train_data, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            a, p, n = batch_data
            loss = model(a, p, n)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        logger.info("epoch 平均 loss：%f" % np.mean(train_loss))
        #测试
        evaluator.eval(epoch)
        #每5轮保存一次权重
        # if epoch % 20 == 0:
    model_path = os.path.join(config["model_path"], "tripletLossModel.pth")
    torch.save(model.state_dict(), model_path)
    return


if __name__ == "__main__":
    main(Config)
    predictor = Predictor(Config)
    inputs = ["密码忘了", "彩信", "宽带坏了", "查询积分","我想查话费"]
    results = predictor.predict(inputs)
    for result in results:
        input, label, label_id = result
        print('输入:', input, '-->预测:', label, '类型的id:', label_id)
