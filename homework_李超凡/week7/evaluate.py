# -*- coding: utf-8 -*-
import torch
from loader import load_data

class Evaluator:
    def __init__(self,config,model,logger):
        self.config=config
        self.model=model
        self.logger=logger
        self.valid_data=load_data(self.config["valid_data_path"],config)
        self.state_dic={"correct":0,"wrong":0}

    def eval(self,epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.state_dic={"correct":0,"wrong":0}
        for index,batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data=[b.cuda() for b in batch_data]
            input_ids,labels=batch_data
            with torch.no_grad():
                pred_result=self.model(input_ids)
            self.write_stats(labels,pred_result)
        acc=self.show_stats()
        return acc

    def write_stats(self,labels,pred_result):
        assert  len(labels)==len(pred_result)
        for true_label,pred_label in zip(labels,pred_result):
            pred_label=torch.argmax(pred_label)
            if int(true_label)==int(pred_label):
                self.state_dic["correct"]+=1
            else:
                self.state_dic["wrong"]+=1
        return

    def show_stats(self):
        correct=self.state_dic["correct"]
        wrong=self.state_dic["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)

