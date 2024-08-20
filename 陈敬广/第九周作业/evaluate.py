import torch

from loader import load_data

'''
模型效果预测

除了训练数据，其他都需要和训练时的环境一致
'''


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.model.eval()
        self.logger = logger
        # 加载测试数据
        self.valid_data = load_data('ner_data/test.txt', config, shuffle=False)

    def eval(self, epoch):
        self.logger.info('第%d轮模型预测开始' % epoch)
        # 记录模型预测情况
        self.stats_dict = {'correct': 0, 'wrong': 0}

        for index, batch_data in enumerate(self.valid_data):
            input_seqs, label_seqs = batch_data
            with torch.no_grad():
                y_pred = self.model.forward(input_seqs)
                y_pred = torch.argmax(y_pred, dim=-1)
                # y_pred = y_pred.cpu().numpy()
                self.write_stats(y_pred, label_seqs)
        self.show_stats(epoch)

    def write_stats(self, y_pred, label_seqs):
        for y_ps, y_ts in zip(y_pred, label_seqs):
            for y_p,y_t in zip(y_ps, y_ts):
                if y_p == y_t:
                    self.stats_dict['correct'] += 1
                else:
                    self.stats_dict['wrong'] += 1

    def show_stats(self, epoch):
        acc = self.stats_dict['correct'] / (self.stats_dict['correct'] + self.stats_dict['wrong'])
        self.logger.info('第%d轮准确率:%f' % (epoch, acc))
