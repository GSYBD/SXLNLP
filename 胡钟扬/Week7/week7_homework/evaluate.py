# -*- coding: utf-8 -*-
import torch
from loader import load_data

import time


class Evaluator(object):
    def __init__(self,config, model, logger):
        super(Evaluator, self).__init__()
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct":0, "wrong":0, "avg_example_len":0, "prediction_time": 0}  #用于存储测试结果
    
    
    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        
        self.stats_dict = {"correct": 0, "wrong": 0, "avg_example_len":0, "prediction_time": 0}  # 清空上一轮结果
        
        self.model.eval()
        
        self.total_input_len = 0 # 所有样本的总长度
        self.total_input_num = 0
        
        start_time = time.time()
        for index, batch in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch = [d.cuda() for d in batch]
            
            len_batch = len(batch)
            
            input_ids, labels = batch
            
            for example in input_ids:
                self.total_input_len+=len(example)
                self.total_input_num += 1
            
            with torch.no_grad():
                y_pred= self.model(input_ids)
            self.write_stats(labels, y_pred)
        
        end_time = time.time()
        
        # 记录推理时间
        self.stats_dict["prediction_time"] = end_time - start_time
        # 记录平均样本长度
        self.stats_dict["avg_example_len"] = self.total_input_len/self.total_input_num
        
        acc = self.show_stats()
        
        
        return acc
    
    
    def write_stats(self, labels, y_pred):
        assert len(labels) == len(y_pred)
        correct, wrong = (0, 0)
        
        for label, pred in zip(labels, y_pred):
            pred= torch.argmax(pred)
            
            if int(label) == int(pred):
                correct +=1
            else:
                wrong+=1
                
        self.stats_dict["correct"] = correct
        self.stats_dict["wrong"] = wrong
        return 
    
    
    
    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("样本平均长度：%f" % (self.stats_dict['avg_example_len']))
        self.logger.info("预测时长: %f" % (self.stats_dict['prediction_time']))
        self.logger.info("--------------------")
        
        acc = correct / (correct + wrong)
        speed = self.stats_dict['prediction_time']
        return acc