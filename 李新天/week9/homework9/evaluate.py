# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.schema = self.valid_data.dataset.schema
        self.index_to_label = dict((y, x) for x, y in self.schema.items())

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = dict(zip(self.schema.keys(), [defaultdict(int) for i in range(len(self.schema))]))
        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[index * self.config["batch_size"]: (index+1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_id) #不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results, sentences)
        self.show_stats()
        return

    def write_stats(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results) == len(sentences), print(len(labels), len(pred_results), len(sentences))
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()[:len(sentence)]
            true_label = true_label.cpu().detach().tolist()[:len(sentence)]
            for pred, gold in zip(pred_label, true_label):
                key = self.index_to_label[gold]
                self.stats_dict[key]["correct"] += 1 if pred == gold else 0
                self.stats_dict[key]["total"] += 1
        return

    def show_stats(self):
        total = []
        for key in self.schema:
            acc = self.stats_dict[key]["correct"] / (1e-5 + self.stats_dict[key]["total"])
            self.logger.info("符号%s预测准确率：%f"%(key, acc))
            total.append(acc)
        self.logger.info("平均acc：%f" % np.mean(total))
        self.logger.info("--------------------")
        return
