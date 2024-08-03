# -*- coding: utf-8 -*-
import os.path
import csv
import torch
import random
import numpy as np
from config import Config
from loader import load_data
import logging
from model import TorchModel ,choose_optimizer
from evaluate import Evaluator
import time

logging.basicConfig(level=logging.INFO,format = '%(asctime)s = %(name)s -%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
"""
模型训练主程序
"""
import logging


seed = Config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    if not os.path.isdir(config["model_path"]):
            os.mkdir(config["model_path"])
    print("创建文件:output")
    train_data = load_data(config["train_data_path"], config)
    model = TorchModel(config)
    cuda_flag = torch.cuda.is_available()
    print("cuda_flag:",cuda_flag)
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    optimizer = choose_optimizer(config,model)

    evaluator = Evaluator(config,model,logger)

    for epoch in range(config["epoch"]):
        # 记录开始时间
        start_time = time.time()
        epoch = epoch + 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index ,batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids,labels = batch_data

            loss = model(input_ids,labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f " % loss)

        logger.info("epoch average loss:%f" %np.mean(train_loss))
        acc = evaluator.eval(epoch)
        # 记录结束时间
        end_time = time.time()
        # 计算总耗时
        execution_time = end_time - start_time
        with open(csv_filename,mode='a',newline='') as file:
            writer = csv.writer(file)
            writer.writerow([Config["model_type"], Config["learning_rate"], Config["hidden_size"], Config["batch_size"], Config["pooling_style"], epoch, acc, execution_time ])
        # model_path = os.path.join(config["model_path"],"model_%s_lr_%f_hiddensize_%d_batch_%d_polling_%s_epoch_%d.pth" % (Config["model_type"], Config["learning_rate"], Config["hidden_size"], Config["batch_size"], Config["pooling_style"],epoch))
        # torch.save(model.state_dict(),model_path)
    # model_path = os.path.join(config["model_path"],"model_%s_lr_%f_hiddensize_%d_batch_%d_polling_%s.pth" % (Config["model_type"], Config["learning_rate"], Config["hidden_size"], Config["batch_size"], Config["pooling_style"]))
    # torch.save(model.state_dict(),model_path)


    return acc


if __name__ == '__main__':
    # main(Config)
    csv_filename = 'training_results.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model', 'lr', 'hidden_size', 'batch_size', 'pooling_style','epoch', 'acc','execution_time'])
    for model in ["rnn","lstm","bert","gated_cnn","bert_cnn"]: # ,"bert","lstm"
        Config["model_type"] = model
        for lr in [1e-3, 1e-4, 1e-5]:
            Config["learning_rate"] = lr
            for hidden_size in [128,768]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg","max"]:  # "max"
                        Config["pooling_style"] = pooling_style
                        main(Config)



    # for model in ["rnn","lstm","bert","gated_cnn","bert_cnn"]: # ,"bert","lstm"
    #     Config["model_type"] = model
    #     for lr in [1e-3, 1e-4, 1e-5]:
    #         Config["learning_rate"] = lr
    #         for hidden_size in [128,768]:
    #             Config["hidden_size"] = hidden_size
    #             for batch_size in [64, 128]:
    #                 Config["batch_size"] = batch_size
    #                 for pooling_style in ["avg","max"]:  # "max"
    #                     Config["pooling_style"] = pooling_style
    #                     print("最后一轮准确率：", main(Config), "当前配置：", Config)  # 打开一个新的文件里面，把每轮的配置都记下来，输出csv
    #                     # 在每次训练结束后将信息写入CSV文件
    #                     acc = main(Config)
    #                     with open(csv_filename, mode='a', newline='') as file:
    #                         writer = csv.writer(file)
    #
    #                         writer.writerow([Config["model_type"], Config["learning_rate"], Config["hidden_size"], Config["batch_size"], Config["pooling_style"], acc])
