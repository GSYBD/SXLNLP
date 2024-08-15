# -*- coding: utf-8 -*-
import torch
from loader import load_data

"""
模型效果测试
"""

class Evaluator:
    def __init__(self,config,model,logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"],config,shuffle=False)
        self.start_dict = {"correct":0,"wrong":0}

    def eval(self,epoch):
        self.logger.info("开始训练第%d轮模型效果："% epoch)
        self.model.eval()
        self.start_dict = {"correct":0,"wrong":0}
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_idx,labels = batch_data
            with torch.no_grad():
                pred_results = self.model(input_idx)
            self.write_stats(labels, pred_results)
        acc = self.show_stats()
        return  acc



    def write_stats(self,labels,pred_results):
        assert len(labels) == len(pred_results)
        for true_label, pred_label in zip(labels, pred_results):
            pred_label = torch.argmax(pred_label)
            if int(true_label) == int(pred_label):
                self.start_dict['correct'] += 1
            else:
                self.start_dict["wrong"] += 1
        return
    def show_stats(self):
        correct = self.start_dict["correct"]
        wrong = self.start_dict["wrong"]
        self.logger.info("预测集合条目：%d" %(correct + wrong))
        self.logger.info("预测正确条目：%d,预测错误条目：%d" %(correct,wrong) )
        self.logger.info("预测准确率：%f" % (correct/(correct + wrong)))
        self.logger.info("-----------------------")
        return correct/(correct + wrong)


