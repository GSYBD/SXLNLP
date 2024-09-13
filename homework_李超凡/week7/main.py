# -*- coding: utf-8 -*-
import os.path

import numpy as np
import pandas as pd

from config import Config
import torch
import logging
import random
from loader import load_data
from model import TorchModel, choose_optimizer
from evaluate import Evaluator

# 记录训练日志，[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("training.log"),  # 将日志保存到文件
])
logger = logging.getLogger(__name__)

# 设置随机种子
seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # 模型保存目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel(config)
    # 判断GPU是否可用
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("设备GPU可用，迁移模型至GPU")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载模型训练效果
    evaluator = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epoch"]):
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in train_data]
            optimizer.zero_grad()
            input_ids, labels = batch_data
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
    return acc


if __name__ == "__main__":
    train_infor = []
    # for model in ["fast_text", "lstm", "gru", "rnn", "cnn", "gated_cnn", "stack_gated_cnn", "rcnn", "bert", "bert_lstm",
    #               "bert_cnn", "bert_mid_layer", "gated_cnn", ]:
    for model in ["bert", "bert_lstm", "bert_cnn", "bert_mid_layer"]:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128, 256]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", "max"]:
                        Config["pooling_style"] = pooling_style
                        print("开始训练参数组合", [model, lr, hidden_size, batch_size, pooling_style])
                        acc = main(Config)
                        train_infor.append([model, lr, hidden_size, batch_size, pooling_style, acc])
                        print("最后一轮准确率：", main(Config), "当前参数组合：", [model, lr, hidden_size, batch_size, pooling_style])
    result_df = pd.DataFrame(np.array(train_infor),
                             columns=["model", "lr", "hidden_size", "batch_size", "pooling_style", "acc"])
    result_df.to_csv("train_result_infor.csv")
