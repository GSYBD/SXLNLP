import torch.cuda

from loader import load_data
import re
import numpy as np
from collections import defaultdict
class Evaluate:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.schema_key = ['LOCATION', 'ORGANIZATION', 'PERSON', 'TIME']
        self.valid_data = load_data(config['valid_data_path'], config)

    def eval(self, epoch):
        self.logger.info('开始验证第%d轮' % epoch)
        self.model.eval()
        self.stats_dict = {key: defaultdict(int) for key in self.schema_key}
        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[index * self.config['max_length']: (index + 1) * self.config['max_length']]
            if torch.cuda.is_available():
                batch_data = [c.cuda() for c in batch_data]
            input_ids, labels = batch_data
            with torch.no_grad():
                pred_res = self.model(input_ids)[0]
            self.write_stats(pred_res, labels, sentences)
        acc = self.show_stats()
        return acc

    def write_stats(self, pred_res, labels, sentences):
        for pred_y, true_y, sentence in zip(pred_res, labels, sentences):
            pred_y = torch.argmax(pred_y, dim=-1)
            pred_y = pred_y.cpu().detach().tolist()
            true_y = true_y.cpu().detach().tolist()
            pred_entities = self.decode(pred_y, sentence)
            true_entities = self.decode(true_y, sentence)
            for key in self.schema_key:
                # 本次预测准确实体
                self.stats_dict[key]['预测准确实体数'] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                # 本次样本实体数
                self.stats_dict[key]['样本实体数'] += len(true_entities[key])
                # 本次预测出来的实体数
                self.stats_dict[key]['预测实体数'] += len(pred_entities[key])


    def show_stats(self):
        F1_SCORES = []
        # 计算每个实体的准确率和召回率
        for key in self.schema_key:
            # 准确率
            p = self.stats_dict[key]['预测准确实体数'] / (1e-5 + self.stats_dict[key]['预测实体数'])
            # 召回率
            recall = self.stats_dict[key]['预测准确实体数'] / (1e-5 + self.stats_dict[key]['样本实体数'])

            F1 = (2 * p * recall) / (1e-5 + p + recall)
            F1_SCORES.append(F1)
            self.logger.info(f"实体{key},准确率:{p},召回率:{recall},F1:{F1}")
        self.logger.info("Macro-F1: %f" % np.mean(F1_SCORES))
        # 计算所有实体的准确率和召回率
        correct_sum = sum([self.stats_dict[key]['预测准确实体数'] for key in self.schema_key])
        true_enti = sum([self.stats_dict[key]['样本实体数'] for key in self.schema_key])
        predict_sum = sum([self.stats_dict[key]['预测实体数'] for key in self.schema_key])
        micro_p = correct_sum / (1e-5 + predict_sum)
        micro_recall = correct_sum / (1e-5 + true_enti)
        micro_F1 = (2 * micro_p * micro_recall) / (1e-5 + micro_p + micro_recall)
        self.logger.info("Micro-F1 %f" % micro_F1)
        self.logger.info("--------------------")
        return

    # 解码，解码出正确的实体，不能以序列标注的准确率当成实体的准确率
    def decode(self, labels, sentence):
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
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







