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
        self.bio_schema = self.valid_data.dataset.bio_schema
        self.attribute_schema = self.valid_data.dataset.attribute_schema
        self.text_data = self.valid_data.dataset.text_data
        self.index_to_label = dict((y, x) for x, y in self.attribute_schema.items())

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"object_acc":0, "attribute_acc": 0, "value_acc": 0, "full_match_acc":0}
        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            text_data = self.text_data[index * self.config["batch_size"]: (index+1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id = batch_data[0]   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                attribute_pred, bio_pred = self.model(input_id) #不输入labels，使用模型当前参数进行预测
            self.write_stats(attribute_pred, bio_pred, text_data)
        self.show_stats()
        return

    def write_stats(self, attribute_pred, bio_pred, text_data):
        attribute_pred = torch.argmax(attribute_pred, dim=-1)
        bio_pred = torch.argmax(bio_pred, dim=-1)
        for attribute_p, bio_p, info in zip(attribute_pred, bio_pred, text_data):
            context, object, attribute, value = info
            bio_p = bio_p.cpu().detach().tolist()
            pred_object, pred_value = self.decode(bio_p, context)
            pred_attribute = self.index_to_label[int(attribute_p)]
            self.stats_dict["object_acc"] += int(pred_object == object)
            self.stats_dict["attribute_acc"] += int(pred_attribute == attribute)
            self.stats_dict["value_acc"] += int(pred_value == value)
            if pred_value == value and pred_attribute == attribute and pred_object == object:
                self.stats_dict["full_match_acc"] += 1
        return

    def decode(self, pred_label, context):
        pred_label = "".join([str(i) for i in pred_label])
        pred_obj = self.seek_pattern("01*", pred_label, context)
        pred_value = self.seek_pattern("23*", pred_label, context)
        return pred_obj, pred_value

    def seek_pattern(self, pattern, pred_label, context):
        pred_obj = re.search(pattern, pred_label)
        if pred_obj:
            s, e = pred_obj.span()
            pred_obj = context[s:e]
        else:
            pred_obj = ""
        return pred_obj

    #打印结果
    def show_stats(self):
        for key, value in self.stats_dict.items():
            self.logger.info("%s : %s " %(key, value / len(self.text_data)))
        self.logger.info("--------------------")
        return
