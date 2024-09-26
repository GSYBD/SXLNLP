import logging
import os

import numpy as np
import torch
from evaluate import Evaluator
from loader import load_data
from model import NerModel, choose_optim

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(config):
    # (1)创建模型参数保存路径
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # (2)加载训练数据
    train_data = load_data(config['train_data_path'], config)

    # (3)创建模型对象
    model = NerModel(config)
    tuning_tactics = config["tuning_tactics"]


    # (4)选择优化器
    optim = choose_optim(config, model)

    # (5)GPU使用
    if torch.cuda.is_available():
        model = model.cuda()
    # (6) 加载模型效果测试对象
    evaluator = Evaluator(config,model,logger)
    # (7)开始训练
    for epoch in range(config['epoch']):
        epoch += 1
        logger.info('第%d轮模型训练开始' % epoch)
        # 记录本轮损失函数值
        watch_loss = []

        for batch_data in train_data:
            # (1)训练数据
            input_seqs,label_seqs = batch_data
            # (2)前向计算loss,记录loss
            loss = model.forward(input_seqs, label_seqs)
            watch_loss.append(loss.item())
            # (3) 梯度反向传播
            loss.backward()
            # (4) 梯度更新
            optim.step()
            # (5) 梯度归零
            optim.zero_grad()
        logger.info('第%d轮平均loss:%f' % (epoch,np.mean(watch_loss)))
        # (6) 模型效果预测
        evaluator.eval(epoch)
    # (8) 保存模型参数
    model_path = os.path.join(config["model_path"], "%s.pth" % tuning_tactics)
    save_tunable_parameters(model, model_path)  #保存模型权重
    return

def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)

if __name__ == '__main__':
    from config import Config
    main(Config)
