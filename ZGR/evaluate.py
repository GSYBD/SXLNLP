# -*- coding: utf-8 -*-
import torch
import numpy as np
import re
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

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            input_ids, attention_mask, labels = batch_data
            if torch.cuda.is_available():
                input_ids, attention_mask, labels = [d.cuda() for d in [input_ids, attention_mask, labels]]
            with torch.no_grad():
                pred_results = self.model(input_ids, attention_mask)
            self.write_stats(labels, pred_results, index)
        self.show_stats()
        return

    def write_stats(self, labels, pred_results, index):
        assert len(labels) == len(pred_results)
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
        for true_label, pred_label in zip(labels, pred_results):
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            true_entities = self.decode(true_label)
            pred_entities = self.decode(pred_label)
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    def show_stats(self):
        F1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return

    def decode(self, labels):
        results = defaultdict(list)
        labels = labels.cpu().detach().tolist()
        for i, label in enumerate(labels):
            if label == 0:  # O
                continue
            # Example label mapping: B-LOCATION: 0, I-LOCATION: 1
            if label == 0:  # B-LOCATION
                results["LOCATION"].append(i)
            elif label == 1:  # I-LOCATION
                results["LOCATION"].append(i)
            # Add similar cases for other labels
        return results
