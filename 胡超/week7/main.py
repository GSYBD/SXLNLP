# -*- coding: utf-8 -*-
"""
author: Chris Hu
date: 2024/8/1
desc:
sample
"""

import torch
import random
import os
import numpy as np
from loguru import logger
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from week7.utils import preprocess_data_and_config, excel_report, pretty_excel
from datetime import datetime
import time

"""
模型训练主程序
"""

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # 创建保存模型的目录
    if not os.path.exists(config["model_path"]):
        os.makedirs(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 训练
    acc = None
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)

        model_path = os.path.join(config["model_path"],
                                  f"{config['model_type']}_bs{config['batch_size']}"
                                  f"_lr{config['learning_rate']}_pool_{config['pooling_style']}_epoch{epoch}.pth")
        if config.get("save_model", False):
            torch.save(model.state_dict(), model_path)  # 保存模型权重
            config['model_full_path'] = model_path
    return acc


def compare_models():
    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    logger.remove()
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    Config["model_path"] = f'output/{now}'
    logger.add(f"./logs/test_model_{now}.log", level="INFO")
    report_path = os.path.join('./reports', f"training_report_{now}.xlsx")
    Config['save_model'] = True
    for model in ["lstm", "gated_cnn", "bert", "bert_cnn"]:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", "mean"]:
                        Config["pooling_style"] = pooling_style
                        start = time.perf_counter()
                        last_acc = main(Config)
                        end = time.perf_counter()
                        Config['acc'] = f"{last_acc:.5f}"
                        Config['time(s)'] = f"{end - start:.5f}"
                        print("最后一轮准确率：", last_acc, "当前配置：", Config)
                        excel_report(Config, report_path)
    pretty_excel(report_path)


if __name__ == "__main__":
    # preprocess_data_and_config(r'./data/文本分类练习.csv')
    # main(Config)

    compare_models()
