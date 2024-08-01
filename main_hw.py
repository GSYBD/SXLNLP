# -*- coding: utf-8 -*-

import torch
import random
import os
import numpy as np
import logging
import csv
from config_hw import Config
from model_hw import TorchModel, choose_optimizer
from evaluate_hw import Evaluator
from loader_hw import load_data

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
                pass
               # logger.info("batch loss %f" % loss)
        # logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
        m = config["model_type"]
    model_path = os.path.join(config["model_path"], "epoch_%s.pth" % m) #保存训练好的模型
    torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc

# 使用训练好的模型做预测
def predict(model_path, input_vec,config):
    input_size = config["hidden_size"]
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())
    test_data = load_data(config["test_data_path"], config)
    evaluator = Evaluator(config, model, logger)
    for epoch in range(config["epoch"]):
        epoch += 1
        model.eval()  # 测试模式
        with torch.no_grad():  # 不计算梯度
            for index, batch_data in enumerate(test_data):
                # if cuda_flag:
                #     batch_data = [d.cuda() for d in batch_data]
                input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况

                if index % int(len(test_data) / 2) == 0:
                    pass
                model.forward(input_ids, target=labels)  # 模型预测
            pred_acc = evaluator.eval(epoch)
    return pred_acc

if __name__ == "__main__":
    # main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    results = []#记录每轮结果
    for model in ["gated_cnn","rnn","cnn"]:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]: #试两个不同的学习率
            Config["learning_rate"] = lr
            for hidden_size in [128]: #bert的hidden_size改不了，已经在config文件中设置
                Config["hidden_size"] = hidden_size
                for batch_size in [64,128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg"]:
                        Config["pooling_style"] = pooling_style
                        Config['acc准确率'] = main(Config)
                        Config_r = dict([(key, Config[key]) for key in ['model_type','learning_rate','hidden_size','batch_size','pooling_style','acc准确率']])
                        print('con',Config_r)
                        results.append(Config_r) #把每一轮结果记录进列表
                        # print("最后一轮准确率：", main(Config), "当前配置：", Config_r)
     #把每轮的配置记下来，然后每轮的准确率记下来，每轮准确率可以由main函数来输出
     #每行相对于一组实验，每列相当于这组实验当前选择的参数，最后一列是准确率
    #作业首先要对数据进行训练集跟验证集的切分
    #主要修改loader，加载数据的部分

    #记录训练结果
    header = list(Config_r.keys())
    with open('results.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)  # 提前预览列名，当下面代码写入数据时，会将其一一对应。
        writer.writeheader()  # 写入列名
        writer.writerows(results) #写入结果


