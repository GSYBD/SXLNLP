from loader import load_data
import torch

"""

    模型效果测试

"""


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.stats_dict = {'correct': 0, 'wrong': 0}
        self.valid_data = load_data(config['valid_data_path'], config, shuffle=False)

    def eval(self, epoch):
        self.logger.info('开始测试第%d轮模型效果: ' % epoch)
        self.model.eval()
        self.stats_dict = {'correct': 0, 'wrong': 0}  # 清空上一轮结果
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            sen_to_indices, labels = batch_data
            with torch.no_grad():
                pred_results = self.model(sen_to_indices)
            self.write_stats(labels, pred_results)
        acc = self.show_stats()
        return acc

    def write_stats(self, labels, pred_results):
        assert len(labels) == len(pred_results)
        for true_label, pred_label in zip(labels, pred_results):
            if 0.5 <= int(true_label) + float(pred_label) < 1.5:
                self.stats_dict['wrong'] += 1
            else:
                self.stats_dict['correct'] += 1

    def show_stats(self):
        correct = self.stats_dict['correct']
        wrong = self.stats_dict['wrong']
        self.logger.info('预测集合条目总量: %d' % len(self.valid_data.dataset))
        self.logger.info('预测正确条目: %d, 预测错误条目: %d' % (correct, wrong))
        self.logger.info('预测准确率: %f' % (correct / len(self.valid_data.dataset)))
        self.logger.info('------------------------')
        return correct / len(self.valid_data.dataset)
