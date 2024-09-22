"""
模型效果测试
"""
import re
from collections import defaultdict

import numpy as np
import torch
from loader import load_data_batch


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        # 选择验证集合
        self.dataset = load_data_batch(config["valid_data_path"], config, shuffle=False)
        self.schema = self.dataset.dataset.schema
        self.index_to_label = dict((y, x) for x, y in self.schema.items())

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = dict(zip(self.schema.keys(), [defaultdict(int) for i in range(len(self.schema))]))
        self.model.eval()
        for index, batch_data in enumerate(self.dataset):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            # 句子 batch_size 化
            sentence = self.dataset.dataset.sentence_list[
                       index * self.config["batch_size"]: (index + 1) * self.config["batch_size"]]
            input_id, labels = batch_data
            with torch.no_grad():
                pred_results = self.model(input_id)
                self.write_stats(labels, pred_results, sentence)
            self.show_stats()
        return

    # 计算准确率的
    def write_stats(self, ture_labels, pred_results, sentences):
        assert len(ture_labels) == len(pred_results) == len(sentences), print(len(ture_labels), len(pred_results), len(sentences))
        if not self.config["use_crf"]:
            # 获取最大下标值
            pred_results = torch.argmax(pred_results, dim=-1)
        for true_label, pred_result, sentence in zip(ture_labels, pred_results, sentences):
            if not self.config["use_crf"]:
                # 和sentence 等长
                pred_result = pred_result.cpu().detach().tolist()[:len(sentence)]
            # 和sentence 等长
            true_label = true_label.cpu().detach().tolist()[:len(sentence)]
            for pred, index in zip(pred_result, true_label):
                if index == -1:
                    continue
                key = self.index_to_label[index]
                self.stats_dict[key]["correct"] += 1 if pred == index else 0
                self.stats_dict[key]["total"] += 1
        return

    # 显示准确率
    def show_stats(self):
        total = []
        for key in self.schema:
            acc = self.stats_dict[key]["correct"] / (1e-5 + self.stats_dict[key]["total"])
            self.logger.info("符号%s预测准确率：%f" % (key, acc))
            total.append(acc)
        self.logger.info("平均acc：%f" % np.mean(total))
        self.logger.info("--------------------")
        return
