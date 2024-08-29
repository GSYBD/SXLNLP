# -*- coding: utf-8 -*-
import logging
import os

import torch
import numpy as np
from transformers import BertTokenizer

from config import Config
from evaluate import XEvaluate
from model import XModel
from loader import XFileLoader

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    epoch_num = Config.get("epoch_num") #训练轮数
    # batch_size = Config.get("batch_size") # 每批的数据量
    # hidden_size = Config.get("hidden_size") # 隐藏层大小其实可以随便写，只要最后一层维度正确即可
    # embedding_dim = Config.get("embedding_dim") # 词向量维度
    # learning_rate = Config.get("learning_rate") # 学习率

    model_path_save = Config["model_path_save"] # 保存模型位置,为空不保存

    tokenizer = BertTokenizer.from_pretrained(Config["bert_model_path"])
    dataLoader = XFileLoader(Config,tokenizer)
    #加载训练数据
    dataLoader.load_data()
    train_data = dataLoader.build_train_data()

    #加载模型效果测试数据
    xEvaluate = XEvaluate(Config)


    #定义模型
    model = XModel(Config)
    #优化器
    optim = choose_optimizer(Config,model)
    for epoch in range(epoch_num):
        epoch += 1
        model.train()
        watch_loss = []
        logger.info("epoch %d begin" % epoch)

        for index, batch_data in enumerate(train_data):
            optim.zero_grad()
            x, y,mask = batch_data
            loss = model.forward(x, y, mask)  # 计算损失
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        logger.info("第%d轮平均loss:%f" % (epoch, np.mean(watch_loss)))
        logger.info("测试本轮模型效果")
        _,pred_char = xEvaluate.generate_sentence("晚安魔都夜景",model,tokenizer,100)
        print(pred_char)
        _,pred_char = xEvaluate.generate_sentence("比尔盖茨是谁",model,tokenizer,100)
        print(pred_char)
        # 保存模型
        if model_path_save != "":
            save_path = os.path.join(os.getcwd()+"/"+model_path_save, "epoch_%d.pth" % epoch)
            torch.save(model.state_dict(), save_path)

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)

if __name__ == '__main__':
    main()