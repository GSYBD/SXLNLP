import torch.cuda

from loader import load_data, load_schema
from collections import defaultdict
import numpy as np


class Evaluate:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config['valid_data_path'], config, shuffle=False)
        self.schema = load_schema(config['schema_path'])
        self.stats_dict = {x: {'correct': 0, 'total': 0} for x in self.schema}
        self.index_to_schema = {y: x for x, y in self.schema.items()}

    def eval(self, epoch):
        self.logger.info("开始验证第%d轮" % epoch)
        self.model.eval()
        self.stats_dict = {x: defaultdict(int) for x in self.schema}
        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[
                        index * self.config['max_length']: (index + 1) * self.config['max_length']]
            if torch.cuda.is_available():
                batch_data = [c.cuda() for c in batch_data]
            input_ids, labels = batch_data
            with torch.no_grad():
                pred_res = self.model(input_ids)
            self.write_stats(pred_res, labels, sentences)
        acc = self.show_stats()
        return acc

    def write_stats(self, pred_res, labels, sentences):
        pred_res = torch.argmax(pred_res, dim=-1)
        for true_y, pred_y, sentence in zip(labels[:len(sentences)], pred_res[:len(sentences)], sentences):
            for t, p in zip(true_y, pred_y):
                if int(t) == -1:
                    continue
                key = self.index_to_schema[int(t)]
                self.stats_dict[key]['correct'] += 1 if int(t) == int(p) else 0
                self.stats_dict[key]['total'] += 1

    def show_stats(self):
        total_sum = []
        for key in self.schema:
            correct = self.stats_dict[key]['correct']
            total = self.stats_dict[key]['total']
            self.logger.info(f"符号{key}预测准确率为:{correct / total}")
            total_sum.append(correct / total)
        self.logger.info("平均准确率为%f" % (np.mean(total_sum)))
        self.logger.info("--------------------")
