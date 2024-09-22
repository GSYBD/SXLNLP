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
        self.valid_data = load_data(config['valid_data_path'], config, shuffle=False)
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}

    def eval(self, epoch):
        self.logger.info('开始测试第%d轮模型效果' % epoch)
        self.model.eval()
        for idx, batch_data in enumerate(self.valid_data):
            sens = self.valid_data.dataset.sentences[
                   idx * self.config['batch_size']: (idx + 1) * self.config['batch_size']]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                predict_results = self.model(input_id)
            self.write_stats(labels, predict_results, sens)
        self.show_stats()

    def write_stats(self, labels, predict_results, sentences):
        assert len(labels) == len(predict_results) == len(sentences)
        if not self.config['use_crf']:
            predict_results = torch.argmax(predict_results, dim=-1)
        for true_label, predict_label, sentence in zip(labels, predict_results, sentences):
            if not self.config["use_crf"]:
                """
                .cpu() 方法用于将Tensor对象从GPU内存转移到CPU内存
                .detach() 方法用于分离Tensor的计算图
                """
                predict_label = predict_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            true_entities = decode(sentence, true_label)
            predict_entities = decode(sentence, predict_label)
            print('------------------------')
            print(true_entities)
            print(predict_entities)
            print('------------------------')
            """
            正确率 = 识别出的正确实体数 / 识别出的实体数
            召回率 = 识别出的正确实体数 / 样本的实体数
            """
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len([ent for ent in predict_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(predict_entities[key])

    def show_stats(self):
        f1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            f1 = (2 * precision * recall) / (precision + recall + 1e-5)
            f1_scores.append(f1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, f1))
        self.logger.info("Macro-F1: %f" % np.mean(f1_scores))
        correct_predict = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_predict = sum([self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_entity = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_predict / (total_predict + 1e-5)
        micro_recall = correct_predict / (true_entity + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")


def decode(sentence, labels):
    sentence = '$' + sentence
    # labels经过补零, 所以截取至句子长度, 将每句话的labels化为字符串
    labels = ''.join(str(x) for x in labels[:len(sentence) + 1])
    results = defaultdict(list)
    for location in re.finditer('(04+)', labels):  # 返回迭代器
        s, e = location.span()  # 返回起始、结束位置
        results['LOCATION'].append(sentence[s: e])
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
