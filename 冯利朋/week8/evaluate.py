from loader import load_data
import torch


class Evaluate:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.train_data = load_data(config['train_data_path'], config)
        self.valid_data = load_data(config['valid_data_path'], config)
        self.stats_dict = {'correct': 0, 'wrong': 0}

    # 加载训练集数据，并做预测，用于匹配验证集数据
    def load_know(self):
        self.question_index_to_stander_index = {}
        self.question_ids = []
        for target, questions in self.train_data.dataset.know.items():
            for question in questions:
                self.question_index_to_stander_index[len(self.question_ids)] = target
                self.question_ids.append(question)
        self.question_ids = torch.stack(self.question_ids, dim=0)
        self.model.eval()
        with torch.no_grad():
            if torch.cuda.is_available():
                self.question_ids = self.question_ids.cuda()
            self.know_vectors = self.model(self.question_ids)
            self.know_vectors = torch.nn.functional.normalize(self.know_vectors, dim=-1)

    def eval(self, epoch):
        self.logger.info('开始验证第%d轮' % epoch)
        self.stats_dict = {'correct': 0, 'wrong': 0}
        self.model.eval()
        self.load_know()
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
        for pred_y, true_y in zip(pred_res, labels):
            res = torch.mm(pred_y.unsqueeze(0), self.know_vectors.T)
            hit_index = int(torch.argmax(res))
            hit_index = self.question_index_to_stander_index[hit_index]
            if int(hit_index) == int(true_y):
                self.stats_dict['correct'] += 1
            else:
                self.stats_dict['wrong'] += 1

    def show_stats(self):
        correct = self.stats_dict['correct']
        wrong = self.stats_dict['wrong']
        total = correct + wrong
        self.logger.info("本次一共验证%d条数据,正确%d条,错误%d条" % (total, correct, wrong))
        self.logger.info("本次正确率为%f" % (correct / total))
        return correct / total
