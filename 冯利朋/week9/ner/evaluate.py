from loader import load_data
from collections import defaultdict
import torch
import re
import numpy as np
class Evaluate:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config['valid_data_path'], config, shuffle=False)
        self.schema = ['LOCATION', 'ORGANIZATION', 'PERSON', 'TIME']
        self.stats_dict = {s: defaultdict(int) for s in self.schema}

    def eval(self, epoch):
        self.logger.info("开始验证第%d轮" % epoch)
        self.model.eval()
        self.stats_dict = {s: defaultdict(int) for s in self.schema}
        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[index * self.config['batch_size']: (index + 1) * self.config['batch_size']]
            if torch.cuda.is_available():
                batch_data = [c.cuda() for c in batch_data]
            input_ids, labels = batch_data
            with torch.no_grad():
                pred_res = self.model(input_ids)
            self.write_stats(pred_res, labels, sentences)
        acc = self.show_stats()
        return acc
    def write_stats(self,pred_res, labels, sentences):
        pred_res = torch.argmax(pred_res, dim=-1)
        for true_y, pred_y, sentence in zip(labels, pred_res, sentences):
            true_y = true_y.detach().cpu().tolist()
            pred_y = pred_y.detach().cpu().tolist()
            true_entities = self.decode(true_y, sentence)
            pred_entities = self.decode(pred_y, sentence)
            for key in self.schema:
                # 预测正确数
                self.stats_dict[key]['预测正确数'] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                # 样本实体数
                self.stats_dict[key]['样本实体数'] += len(true_entities[key])
                # 预测实体数
                self.stats_dict[key]['预测实体数'] += len(pred_entities[key])


    def show_stats(self):
        F1_scores = []
        for key in self.schema:
            # 准确率
            pred_c = self.stats_dict[key]['预测正确数'] / (1e-5 + self.stats_dict[key]['预测实体数'])
            # 召回率
            recall = self.stats_dict[key]['预测正确数'] / (1e-5 + self.stats_dict[key]['样本实体数'])
            F1 = (2 * pred_c * recall) / (1e-5 + pred_c + recall)
            F1_scores.append(F1)
            self.logger.info(f"{key}准确率:{pred_c},召回率:{recall},F1={F1}")
        self.logger.info(f"micro-F1={np.mean(F1_scores)}")
        pred_c_sum = sum([self.stats_dict[key]['预测正确数'] for key in self.schema])
        true_sum = sum([self.stats_dict[key]['样本实体数'] for key in self.schema])
        pred_sum = sum([self.stats_dict[key]['预测实体数'] for key in self.schema])
        pred_c = pred_c_sum / (1e-5 + pred_sum)
        recall = pred_c_sum / (1e-5 + true_sum)
        micro_f1 = (2 * pred_c * recall) / (pred_c + recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")


    def decode(self, labels, sentence):
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        res = defaultdict(list)
        for locations in re.finditer("(04+)", labels):
            s, e = locations.span()
            res['LOCATION'].append(sentence[s: e])
        for locations in re.finditer("(15+)", labels):
            s, e = locations.span()
            res['ORGANIZATION'].append(sentence[s: e])
        for locations in re.finditer("(26+)", labels):
            s, e = locations.span()
            res['PERSON'].append(sentence[s: e])
        for locations in re.finditer("(37+)", labels):
            s, e = locations.span()
            res['TIME'].append(sentence[s: e])
        return res


