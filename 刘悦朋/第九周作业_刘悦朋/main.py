import logging
import time
import random
import numpy as np
import torch
import os
import json
from config import Config
from model import TorchModel, choose_optimizer
from loader import load_data
from evaluate import Evaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
log_path = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".log"
handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
logger.addHandler(handler)

"""

    模型训练主程序
    
"""

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    logger.info(json.dumps(config, ensure_ascii=False, indent=2))
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    # multi_gpu_flag = False
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        device_ids = list(range(config["num_gpus"]))
        if len(device_ids) > 1:
            logger.info("使用多卡gpu训练")
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            # multi_gpu_flag = True
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config, logger)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    for epoch in range(config['epoch']):
        epoch += 1
        model.train()
        logger.info("Epoch %d / %d" % (epoch, config['epoch']))
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)


if __name__ == "__main__":
    main(Config)
