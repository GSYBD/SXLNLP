import torch.cuda

from loader import load_data
class Evaluate:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config['valid_data_path'], config, shuffle=False)
        self.stats_dict = {'correct': 0, 'wrong': 0}
    def eval(self, epoch):
        self.logger.info("开始验证第%d轮的模型" % epoch)
        self.model.eval()
        self.stats_dict = {'correct': 0, 'wrong': 0}
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [c.cuda() for c in batch_data]
            input_ids, labels = batch_data
            with torch.no_grad():
                pred_res = self.model(input_ids)
            self.write_stats(pred_res, labels)
        acc = self.show_stats()
        return acc
    def write_stats(self, pred_res, labels):
        for true_y, pred_y in zip(labels, pred_res):
            pred_y = torch.argmax(pred_y)
            if int(pred_y) == int(true_y):
                self.stats_dict['correct'] += 1
            else:
                self.stats_dict['wrong'] += 1
    def show_stats(self):
        correct = self.stats_dict['correct']
        wrong = self.stats_dict['wrong']
        self.logger.info("本次一共验证%d条数据,其中正确%d条,错误%d条" % ((correct + wrong), correct, wrong))
        self.logger.info("本次的正确率为%f" % (correct / (correct + wrong)))
        return correct / (correct + wrong)
