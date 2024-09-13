# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
import pandas as pd
import time
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("train.log", encoding='utf-8')])
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
    print(len(train_data))
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
                logger.info("%d batch loss %f" % (index, loss))
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        start_time = time.time()
        acc = evaluator.eval(epoch)
        end_time = time.time()
        eval_time = end_time - start_time
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc, eval_time


if __name__ == "__main__":
    result = []

    # 先试下整体流程，以及bertlstm, hidden_size=256, batch_size=128, 会不会爆显存
    # 结果立刻爆了
    # acc, eval_time = main(Config) 
    # result.append([acc, Config['model_type'], Config['learning_rate'], Config['hidden_size'], Config['batch_size'], Config['pooling_style'], eval_time])
    # print("最后一轮准确率：", acc, "当前配置：", result)

    # result_df = pd.DataFrame(result,
    #                          columns=['acc', "model", "lr", "hidden_size", "batch_size", "pooling_style", 'eval_time'])
    # result_df.to_csv("result.tsv", sep='\t', index=False)
    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    try:
        # for model in ['lstm', 'gru', 'cnn', "gated_cnn"]:
        for model in ['bert', 'bert_lstm', 'bert_cnn']:
            Config["model_type"] = model
            Config["epoch"] = 5  # 这些数据，微调5轮够了
            # for lr in [1e-3, 1e-4]:
            for lr in [1e-5, 5e-5]:  # bert微调,学习率要小一些，大了测试下loss不下降（日志6210-6302行）
                Config["learning_rate"] = lr
                for hidden_size in [64, 128]:
                    Config["hidden_size"] = hidden_size
                    for batch_size in [128, 256]:
                        if 'bert' in model:
                            Config["batch_size"] = int(batch_size / 8)
                        else:
                            Config["batch_size"] = batch_size
                        for pooling_style in ['max', "avg"]:
                            Config["pooling_style"] = pooling_style
                            acc, eval_time = main(Config) 
                            logger.info(f'********\n{acc}\t{Config['model_type']}\t{Config['learning_rate']}\t{Config['hidden_size']}\t{Config['batch_size']}\t{Config['pooling_style']}\t{eval_time}')
                            print("最后一轮准确率：", acc, "当前配置：", Config)
                            result.append([acc, Config['model_type'], Config['learning_rate'], Config['hidden_size'], Config['batch_size'], Config['pooling_style'], eval_time])
        result_df = pd.DataFrame(result,
                                columns=['acc', "model", "lr", "hidden_size", "batch_size", "pooling_style", 'eval_time'])
        # result_df.to_csv("result.tsv", sep='\t', index=False)
        result_df.to_csv("result.tsv", sep='\t', index=False, mode='a')
    except Exception as e:
        print(e)
        for res in result:
            print(res)
