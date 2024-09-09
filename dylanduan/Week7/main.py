# -*- coding: utf-8 -*-

import torch
import os
import random
import csv
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
    train_data = load_data(config["train_data_path"], config)
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
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
        csv_file_path = "model_results.csv"
        add_config_to_csv(config, csv_file_path, acc)
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc

def add_config_to_csv(config, csv_file_path, acc):
    # 定义CSV的列名
    fieldnames = ['model_type', 'max_length', 'hidden_size', 'kernel_size', 'num_layers', 
                  'epoch', 'batch_size', 'pooling_style', 'optimizer', 'learning_rate', 
                  'class_num', 'vocab_size', 'acc']
    # 准备要写入的数据
    data = {
        'model_type': config['model_type'],
        'max_length': config['max_length'],
        'hidden_size': config['hidden_size'],
        'kernel_size': config['kernel_size'],
        'num_layers': config['num_layers'],
        'epoch': config['epoch'],
        'batch_size': config['batch_size'],
        'pooling_style': config['pooling_style'],
        'optimizer': config['optimizer'],
        'learning_rate': config['learning_rate'],
        'class_num': 2,  # 假设为二分类任务
        'vocab_size': 0,  # 这里需要您提供实际的词汇表大小
        'acc': acc
    }
    # 检查文件是否存在
    file_exists = os.path.isfile(csv_file_path)
    # 打开文件，如果不存在则创建
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        # 如果文件不存在，写入列名
        if not file_exists:
            writer.writeheader()
        # 写入数据
        writer.writerow(data)
    print(f"Data has been added to {csv_file_path}")


if __name__ == "__main__":
    main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    for model in ["gated_cnn", "lstm", "gru", "cnn"]:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg"]:
                        Config["pooling_style"] = pooling_style
                        print("最后一轮准确率：", main(Config), "当前配置：", Config)
