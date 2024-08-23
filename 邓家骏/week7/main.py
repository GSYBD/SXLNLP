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
# import multiprocessing
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
    # 存储每epoch的loss
    loss_arr = []
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
        mean_loss = np.mean(train_loss)
        loss_arr.append(mean_loss)
        logger.info("epoch average loss: %f" % mean_loss)
        acc = evaluator.eval(epoch)
        # 不用保存模型了，练手用的，也没多少真实数据，能出个不同维度对比文档就够了。
    # model_path = os.path.join(config["model_path"], f"{config['model_type']}.pth")
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc,loss_arr

def work(model,max_length,hidden_size,kernel_size,pooling_style,optimizer,num_layers,epoch,batch_size,learning_rate):
    global lock

    
if __name__ == "__main__":
    # main(Config)
    # 分多维输出
# "fast_text","lstm","gru","rnn","cnn","gated_cnn","stack_gated_cnn","rcnn","bert","bert_lstm","bert_cnn","bert_mid_layer","bert_diy"
    with open(r'D:\code\data\week7_data\log2.txt','a',encoding='utf8') as f:
        f.write('model,max_length,kernel_size,pooling_style,optimizer,num_layers,epoch,batch_size,learning_rate,acc,loss_arr \n')

    # 多进程计算
    # 算了，没写好，有空再试。
    # "fast_text","lstm","gru","rnn","cnn","gated_cnn","stack_gated_cnn","rcnn",
    for model in ["bert","bert_lstm","bert_cnn","bert_mid_layer","bert_diy"]:
        for max_length in [30] :
            for hidden_size in [64,256,512] :
                if('bert' in model and hidden_size != 64):
                    continue
                for kernel_size in [3,5,7] :
                    if model != 'cnn' and kernel_size != 3:
                        continue
                    for pooling_style in ['avg','max'] :
                        if(model not in ["fast_text",'bert'] and pooling_style != 'avg'):
                            continue
                        for optimizer in ['adam'] :
                            for num_layers in [2] :
                                for epoch in [15,25] :
                                    for batch_size in [64,128] :
                                        for learning_rate in [1e-2,1e-3,1e-5]:
                                            if(model not in ["fast_text","lstm",'bert'] and learning_rate not in [1e-2,1e-3]):
                                                continue
                                            Config["model_type"] = model
                                            Config["hidden_size"] = hidden_size
                                            Config["kernel_size"] = kernel_size
                                            Config["pooling_style"] = pooling_style
                                            Config["optimizer"] = optimizer
                                            Config["num_layers"] = num_layers
                                            Config["epoch"] = epoch
                                            Config["batch_size"] = batch_size
                                            Config["learning_rate"] = learning_rate
                                            # if 'bert' in model :
                                            #     Config['num_hidden_layers'] = num_layers     
                                            acc,loss = main(Config)
                                            s = f"{model},{max_length},{hidden_size},{kernel_size},{pooling_style},{optimizer},{num_layers},{epoch},{batch_size},{learning_rate},{acc},{loss} \n"
                                            with open(r'D:\code\data\week7_data\log.txt','a',encoding='utf8') as f:
                                                    f.write(s)
                                            print(s)
                                               
                                            

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    # for model in ["gated_cnn"]:
    #     Config["model_type"] = model
    #     for lr in [1e-3, 1e-4]:
    #         Config["learning_rate"] = lr
    #         for hidden_size in [128]:
    #             Config["hidden_size"] = hidden_size
    #             for batch_size in [64, 128]:
    #                 Config["batch_size"] = batch_size
    #                 for pooling_style in ["avg"]:
    #                     Config["pooling_style"] = pooling_style
    #                     print("最后一轮准确率：", main(Config), "当前配置：", Config)
