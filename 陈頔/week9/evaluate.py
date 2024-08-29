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


    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            # 提取批次，通过index+1，移动到下一个批次的开始位置
            # 对应loader中定义的句子列表
            sentences = self.valid_data.dataset.sentences[index * self.config["batch_size"]: (index+1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_id) #不输入labels，使用模型当前参数进行预测
                # print('pred_results', pred_results, pred_results.shape)
            # 遍历self.valid_data，这个经过loader转换完变成[input_id, labels]准备了输入编码和真实标签
            # 预测结果pred_results把编码后的句子送入模型，真实标签labels
            # sentences对应loader当前处理的原句（未编码直接存储）
            self.write_stats(labels, pred_results, sentences)
        self.show_stats()
        return

    def write_stats(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results) == len(sentences)
        # 不考虑crf的场景，只有发射矩阵，也就是每个预测类别里找最高的，
        # 比如3个字概率结果是[0.1 0.4 0.3]、[0.3 0.1 0.1] [0.1 0.3 0.6]
        # pred_results是9个概率分布，pred_label中应该是个整数序列，如果从0开始，应该是[1 0 2]，代表每个字所处的类别
        # 因为在loader中已经把label的字母标签转成schema中定义的数字，所以每个字中的概率也是按序基于数字类别
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            print('pred_label', pred_label)
            # 用crf的话，会自动帮你把上面的序列重写，生成的也是[1 0 2]这种序列
            # 这里的true和pred label都是每句话中每个字所属的类别序列
            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)
            print('true_entities', true_entities)
            print('pred_entities', pred_entities)
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    def show_stats(self):
        F1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
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

    '''
    {
      "B-LOCATION": 0,
      "B-ORGANIZATION": 1,
      "B-PERSON": 2,
      "B-TIME": 3,
      "I-LOCATION": 4,
      "I-ORGANIZATION": 5,
      "I-PERSON": 6,
      "I-TIME": 7,
      "O": 8
    }
    '''
    #因为传入sentence是原句（不是经过编码），而label是在loader中经过我们处理的，也就是在label第一位加了8来对应
    #所以为了对应我们在前面loader里加的那一位8，所以需要在这里的原句sentence第一个字符也加个占位符防止跟loader错位
    def decode(self, sentence, labels):
        sentence = "$" + sentence
        labels = "".join([str(x) for x in labels[:len(sentence)+1]])
        results = defaultdict(list)
        # 比如04+代表一个0和若干个4，如上表就是一个一个B-location跟若干个I
        # 在crf中生成的labels序列[一句话中每个字的类别号]中找，如果序列中有0和若干个4，就返回开始和结束的索引
        # 在原句中进行切片，pred和true会传入不同的label，所以对原句的切片也不一样，会输出不一样的结果
        # loader中已经把label的字母标签转成schema中定义的数字，所以这里传入的label序列也就是每个字对应的类别已经是schema中定义的数字
        # 所以一个句子有几个字，序列中就有几个类别，相当于给句子中每个字做8分类，序列中每个数字代表每个字概率最高的类别
        # 在分类任务中，需要确保预测的概率分布能跟定义的顺序匹配，这样才能哪个最大就说他是哪一类
        for location in re.finditer("(04+)", labels):
            # s是匹配开始的索引，e是匹配结束的索引（不包括）
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results