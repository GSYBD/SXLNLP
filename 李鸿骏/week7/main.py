# -*- coding: utf-8 -*-
import time
from collections import defaultdict

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data, DataGenerator
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
output_data = defaultdict(list)
output_data["Model"] = []
output_data["Learning_rate"] = []
output_data["Hidden_size"] = []
output_data["Batch_size"] = []
output_data["Train_Avg_Positive_sample"] = []
output_data["Train_Avg_Negative_sample"] = []
output_data["Valid_Avg_Positive_sample"] = []
output_data["Valid_Avg_Negative_sample"] = []
output_data["Train_Avg_length"] = []
output_data["Valid_Avg_length"] = []
output_data["Last_Accuracy"] = []
output_data["time(预测100条耗时)"] = []


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    dg = DataGenerator(config["train_data_path"], config)
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
    train_negative_sample = 0
    train_positive_sample = 0
    valid_negative_sample = 0
    valid_positive_sample = 0
    train_data_avg_text_len = 0
    valid_data_avg_text_len = 0
    eval_cost_time = 0
    for epoch in range(config["epoch"]):
        # 每轮随机抽取数据进行训练与验证
        train_data, valid_data = load_data(dg, config, shuffle=True)
        train_negative_sample += dg.train_negative_sample
        train_positive_sample += dg.train_positive_sample
        valid_negative_sample += dg.valid_negative_sample
        valid_positive_sample += dg.valid_positive_sample
        train_data_avg_text_len += dg.train_data_avg_text_len
        valid_data_avg_text_len += dg.valid_data_avg_text_len
        evaluator.valid_data = valid_data
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
        eval_start_time = time.time()
        acc = evaluator.eval(epoch)
        eval_end_time = time.time()
        eval_cost_time += eval_end_time - eval_start_time
    output_data["Train_Avg_Positive_sample"].append(train_positive_sample / config["epoch"])
    output_data["Train_Avg_Negative_sample"].append(train_negative_sample / config["epoch"])
    output_data["Valid_Avg_Positive_sample"].append(valid_positive_sample / config["epoch"])
    output_data["Valid_Avg_Negative_sample"].append(valid_negative_sample / config["epoch"])
    output_data["Train_Avg_length"].append(train_data_avg_text_len / config["epoch"])
    output_data["Valid_Avg_length"].append(valid_data_avg_text_len / config["epoch"])
    output_data["time(预测100条耗时)"].append(eval_cost_time / config["epoch"])
   # model_path = os.path.join(config["model_path"], "%s_epoch_%d.pth" % (Config["model_type"],epoch))
    # torch.save(model.state_dict(), model_path)  # 保存模型权重

    return acc


if __name__ == "__main__":
    # main(Config)
    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    # for model in ["gated_cnn", "lstm", "gru", "cnn", "bert", "rnn", "stack_gated_cnn", "rcnn", "bert_lstm", "bert_cnn",
    #               "bert_mid_layer"]:
    for model in ["bert", "lstm", "gru", "cnn"]:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                if model == "bert":
                    Config["hidden_size"] = 768
                else:
                    Config["hidden_size"] = hidden_size
                for batch_size in [64]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg"]:
                        Config["pooling_style"] = pooling_style
                        acc = main(Config)
                        print("最后一轮准确率：", acc, "当前配置：", Config)
                        output_data["Model"].append(model)
                        output_data["Learning_rate"].append(lr)
                        output_data["Hidden_size"].append(hidden_size)
                        output_data["Batch_size"].append(batch_size)
                        output_data["Last_Accuracy"].append(acc)
    pd.DataFrame(output_data).to_csv(r"output\output.csv", index=False)
