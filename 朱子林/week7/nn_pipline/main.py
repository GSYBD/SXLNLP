# -*- coding: utf-8 -*-

import torch
import random
import os
import json
import time
import numpy as np
import logging
import csv
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evalutor
from loader import load_data
from washdata import WashData
from predict import Predictor
#打印日志配置 [DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
write_log = {}
'''
main
 fast_text cnn rnn lstm gru gated_cnn stack_gated_cnn rcnn 这些
 上面那些模型的对比数据放在data文件夹下的train_log.csv文件里面了
"bert", "bert_cnn", "bert_lstm"这些中 bert模型训练了10多个小时都没结束，时间太久了，
所以bert训练只取了100条来训练测试
'''

#设置随机种子，你可以确保每次运行代码时，随机数的生成都是一致的
seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def train(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
    #gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        # logger.info("可以使用gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载评测类
    evaluator = Evalutor(config, model, logger)
    start_time = time.perf_counter()
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        # logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            optimizer.zero_grad()
            input_ids, labels = batch_data
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            # if index % int(len(train_data) / 2) == 0:
            #     logger.info("batch loss %f" % loss)
        # logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
    # 计算运行时长
    end_time = time.perf_counter()
    train_time = end_time - start_time
    # logger.info(f"train代码运行时长: {train_time} 秒")
    model_path = os.path.join(config["model_path"], "%s_model.pth" % config["model_type"])
    torch.save(model.state_dict(), model_path)  #保存模型权重

    return acc, train_time

def write_log_to_cvc():
    fieldnames = ["model_type","learning_rate","hidden_size","epoch","batch_size","acc","train_time","predict_time"]
    # 打开CSV文件，以追加模式（'a'）打开
    with open('data/train_bert_log.csv', mode='a', newline='', encoding='utf-8') as file:
        # 创建CSV写入器
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        # 写入列标题（如果文件中没有列标题）
        if file.tell() == 0:
            writer.writeheader()
        # 写入数据
        writer.writerow(write_log)


if __name__ == "__main__":
    config = Config
    #先清洗数据
    WashData(config)
    # fast_text cnn rnn lstm gru gated_cnn stack_gated_cnn rcnn bert bert_lstm bert_cnn bert_mid_layer
    #"bert", "bert_cnn", "bert_lstm" bert模型训练了10多个小时都没训练成功，时间太久了，只能对比下面这些了
    for model_type in ["bert_mid_layer"]:
        config["model_type"] = model_type
        #训练
        acc, train_time = train(config)
        print("最后一轮准确率：", acc, "当前配置：", model_type, "训练时长：", train_time)
        #预测答案
        with open(config["predict_data_path"], 'r',encoding="utf8") as f:
            predictor = Predictor(config)
            input_list = json.load(f)
            results, predict_time = predictor.predict(input_list)
        write_log["model_type"] = model_type
        write_log["learning_rate"] = config["learning_rate"]
        write_log["hidden_size"] = config["hidden_size"]
        write_log["epoch"] = config["epoch"]
        write_log["batch_size"] = config["batch_size"]
        write_log["acc"] = acc
        write_log["train_time"] = train_time
        write_log["predict_time"] = predict_time
        write_log_to_cvc()