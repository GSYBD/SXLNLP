# -*- coding: utf-8 -*-
import sys
import torch
import os
import numpy as np
import logging
import json
from config import Config  # 从配置文件中导入 Config
from evaluate import Evaluator
from loader import load_data
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    logger.info(json.dumps(config, ensure_ascii=False, indent=2))

    # 加载预训练的模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["pretrained_model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["pretrained_model_name"],
        num_labels=config["num_labels"]
    )

    # 标识是否使用GPU
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU可用，将模型迁移至GPU")
        model = model.cuda()

    # 加载优化器
    optimizer = choose_optimizer(config, model)

    # 加载和处理数据
    input_ids, attention_mask, labels = load_data(config["train_data_path"], tokenizer, config, logger)

    # 加载评估器
    evaluator = Evaluator(config, model, logger)

    # 加载损失函数
    loss_func = torch.nn.CrossEntropyLoss()

    # 开始训练
    for epoch in range(config["epoch"]):
        model.train()
        logger.info(f"Epoch {epoch + 1}/{config['epoch']} 开始")

        if cuda_flag:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        logger.info(f"Epoch {epoch + 1} 训练损失: {loss.item():.4f}")
        evaluator.eval(epoch)

    model_path = os.path.join(config["model_path"], f"epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main(Config)  # 传递 Config 作为参数
