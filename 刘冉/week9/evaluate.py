# coding: utf-8

import torch
import numpy as np
from loader import load_data

'''
模型评估
'''

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        data_path = config["valid_path"]
        self.valid_data = load_data(config, data_path, shuffle=False)
        self.schema = self.valid_data.dataset.schema

    def set_stats_dict(self):
        stats_dict = {}
        for key in self.schema:
            stats_dict[key] = {"right": 0, "total": 0, "predict": 0}
        return stats_dict

    def eval(self, epoch):
        self.logger.info("开始第%d轮模型效果测试：" % epoch)
        # 每次都需要重置
        self.stats_dict = self.set_stats_dict()
        self.model.eval()
        for batch_data in self.valid_data:
            if torch.cuda.is_available():
                batch_data = [b.cuda() for b in batch_data]
            sentences, labels = batch_data
            with torch.no_grad():
                pred_labels = self.model(sentences)
            pred_labels = torch.argmax(pred_labels, dim=-1)
            pred_labels = pred_labels.cpu().detach().tolist()
            labels = labels.cpu().detach().tolist()
            self.write_stats(pred_labels, labels)
        self.show_stats()

    def write_stats(self, pred_labels, labels):
        for pred_label, label in zip(pred_labels, labels,):
            for key in self.stats_dict:
                value = self.schema[key]
                for pred_char, char in zip(pred_label, label):
                    if pred_char == value:
                        self.stats_dict[key]["predict"] += 1
                    if char == value:
                        self.stats_dict[key]["total"] += 1
                        if pred_char == char:
                            self.stats_dict[key]["right"] += 1

    def show_stats(self):
        total = []
        for key in self.stats_dict:
            predict = self.stats_dict[key]["predict"]
            acc = self.stats_dict[key]["right"] / (1e-5 + self.stats_dict[key]["total"])
            self.logger.info("符号[%s]预测了[%d]个,准确率：%f" % (key, predict, acc))
            total.append(acc)
        self.logger.info("整体平均准确率：%f" % np.mean(total))
