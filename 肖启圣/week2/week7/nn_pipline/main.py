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
import pandas as pd

# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    # 创建保存模型的目录
    global train_loss, acc
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
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
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)  # 保存模型权重
    return acc


if __name__ == "__main__":
    # Config["model_type"] = "cnn"
    # # main(Config)
    #
    #
    # columns = ['model', 'learning_rate', 'pooling_style']
    # for model in ["cnn", "rnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    model_type = []
    learning_rate1 = []
    hidden_size1 = []
    batch_size1 = []
    pooling_style1 = []
    epoch1 = []
    loss1 = []
    accuary1 = []
    i = 0
    result = {}
    for model in ["fast_text", "lstm", "gru", "rnn", "cnn", "gated_cnn", "stack_gated_cnn", "rcnn", "bert_lstm",
                  "bert_cnn", "bert"]:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128, 256]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", "max"]:
                        Config["pooling_style"] = pooling_style
                        for epoch in [10, 15]:
                            print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])
                            Config["epoch"] = epoch
                            model_type.append(Config["model_type"])
                            learning_rate1.append(Config["learning_rate"])
                            hidden_size1.append(Config["hidden_size"])
                            batch_size1.append(Config["batch_size"])
                            pooling_style1.append(Config["pooling_style"])
                            epoch1.append(Config["epoch"])
                            loss1.append(np.mean(train_loss))
                            accuary1.append(main(Config))

    result['model_type'] = model_type
    result['learning_rate'] = learning_rate1
    result['hidden_size'] = hidden_size1
    result['batch_size'] = batch_size1
    result['pooling_style'] = pooling_style1
    result['epoch'] = epoch1
    result['loss'] = loss1
    result['accuracy'] = accuary1

    df = pd.DataFrame(result)
    df.to_csv('result.csv', index=False)

                            # print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])
                            # result['model_type'] = Config["model_type"]
                            # result['learning_rate'] = Config["learning_rate"]
                            # result['hidden_size'] = Config["learning_rate"]
                            # result['batch_size'] = Config["batch_size"]
                            # result['pooling_style'] = Config["pooling_style"]
                            # result['epoch'] = Config["epoch"]
                            # result['loss'] = np.mean(train_loss)
                            # result['accuracy'] = main(Config)
                            #
                            # df = pd.DataFrame(result)
                            # if i == 0:
                            #     i += 1
                            #     # 第一次写入时创建新文件
                            #     df.to_csv('output_data.csv', mode='w', index=False)
                            # else:
                            #     # 之后追加数据
                            #     df.to_csv('output_data.csv', mode='a', header=False, index=False)
