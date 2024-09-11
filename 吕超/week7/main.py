# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


seed = Config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    # train_data = load_data(config["train_data_path"], config)
    train_data = load_data(config["my_train_data_path"], config)
    #加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        # lyu: 访问到train_data中的每个元素（在这里称为batch_data），还能同时获取到当前元素的索引(在这里称为index)
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels) # lyu: input_ids(128*30), labels(128*1)是batch_data中的前两个元素
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            # print("len(train_data)", len(train_data))   75
            if index % int(len(train_data) / 2) == 0:  # lyu: 每训练一半数据，输出一次loss
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
        # # lyu: 每轮训练完, 保存模型
        # if acc > config["best_acc"]:
        #     config["best_acc"] = acc
        #     # model_path = os.path.join(config["model_path"], "best.pth")
        #     # torch.save(model.state_dict(), model_path)  #保存模型权重
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc

if __name__ == "__main__":
    # main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    for model in ["lstm", "gated_cnn", "bert"]:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128, 256]:  # lyu:bert默认是768, 该引用版本无法更改
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", "max"]:
                        Config["pooling_style"] = pooling_style
                        # print("最后一轮准确率：", main(Config), "当前配置：", Config)
                        # lyu: 打开一个文件,将结果写到csv表, 当前准确率由main函数返回
                        # with open("result.csv", "a") as f:
                        #     f.write("%s,%f,%f,%d,%s,%f\n" % (model, lr, hidden_size, batch_size, pooling_style, main(Config)))
                        with open("../data/my_result.csv", "a") as f:
                            f.write("%s,%f,%f,%d,%s,%f\n" % (model, lr, hidden_size, batch_size, pooling_style, main(Config)))
