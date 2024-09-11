import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer, SiameseNetwork
from loader import load_data_batch
from evaluate import Evaluator

# [DEBUG, INFO, WARNING, ERROR, CRITICAL]


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""
# 通过设置随机种子来复现上一次的结果（避免随机性）
seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # 保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载数据
    dataset = load_data_batch(config["train_data_path"], config)
    # 加载模型
    model = SiameseNetwork(config)
    # 是否使用gpu
    if torch.cuda.is_available():
        logger.info("gpu可以使用，迁移模型至gpu")
        model.cuda()
    # 选择优化器
    optim = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    for epoch in range(config["epoch_size"]):
        epoch += 1
        logger.info("epoch %d begin" % epoch)
        epoch_loss = []
        # 训练模型
        model.train()
        for batch_data in dataset:
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            # x, y = dataiter
            # 反向传播
            optim.zero_grad()
            s1, s2, s3 = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            # 计算梯度
            loss = model(s1, s2, s3)
            # 梯度更新
            loss.backward()
            # 优化器更新模型
            optim.step()
            # 记录损失
            epoch_loss.append(loss.item())
        logger.info("epoch average loss: %f" % np.mean(epoch_loss))
        # 测试模型效果
        acc = evaluator.eval(epoch)
    # 可以用model_type model_path epoch 三个参数来保存模型
    model_path = os.path.join(config["model_path"], "epoch_%d_%s.pth" % (epoch, config["model_type"]))
    torch.save(model.state_dict(), model_path)  # 保存模型权重
    return


if __name__ == "__main__":
    from config import Config

    Config["train_flag"] = True
    main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
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
    # 可以把输出放入文件中 便于查看
    #                     print("最后一轮准确率：", main(Config), "当前配置：", Config)
