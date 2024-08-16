import torch
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
        # 犹豫效果测试需要训练集当做知识库, 再次加载训练集
        # 事实上可以通过传参把前面加载的训练集传进来更合理, 但是为了主流程代码改动量小, 在这里重新加载一遍
        self.train_data = load_data(config['train_data_path'], config)
        self.stats_dict = {'correct': 0, 'wrong': 0}
        self.question_idx_to_standard_question_idx = {}
        self.question_ids = []

    # 将知识库中的问题向量化, 为匹配做准备
    # 每轮训练的模型参数不一样, 生成的向量也不一样, 所以需要每轮测试重新进行向量化
    def knwb_to_vector(self):
        for standard_question_idx, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                # 记录问题编号到标准问题标号的映射, 用来确认答案是否正确
                self.question_idx_to_standard_question_idx[len(self.question_ids)] = standard_question_idx
                self.question_ids.append(question_id)
        with torch.no_grad():
            question_matrices = torch.stack(self.question_ids, dim=0)
            if torch.cuda.is_available():
                question_matrices = question_matrices.cuda()
            self.knwb_vectors = self.model(question_matrices)
            # 将所有向量都作归一化 v / |v|
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)

    def eval(self, epoch):
        self.logger.info('开始测试第%d轮模型效果: ' % epoch)
        self.stats_dict = {'correct': 0, 'wrong': 0}  # 清空前一轮的测试效果
        self.model.eval()
        self.knwb_to_vector()
        for idx, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data
            with torch.no_grad():
                test_question_vectors = self.model(input_id)
            self.write_stats(test_question_vectors, labels)
        self.show_stats()

    def write_stats(self, test_question_vectors, labels):
        assert len(labels) == len(test_question_vectors)
        for test_question_vector, label in zip(test_question_vectors, labels):
            # 通过一次矩阵乘法, 计算输入问题和知识库中所有问题的相似度
            # test_question_vector shape [vec_size]  knwb_vectors shape = [n, vec_size]
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            hit_index = int(torch.argmax(res.squeeze()))  # 命中问题标号
            hit_index = self.question_idx_to_standard_question_idx[hit_index]
            if int(hit_index) == int(label):
                self.stats_dict['correct'] += 1
            else:
                self.stats_dict['wrong'] += 1

    def show_stats(self):
        correct = self.stats_dict['correct']
        wrong = self.stats_dict['wrong']
        self.logger.info('预测集合条目总量: %d' % (correct + wrong))
        self.logger.info('预测正确条目: %d, 预测错误条目: %d' % (correct, wrong))
        self.logger.info('预测准确率: %f' % (correct / (correct + wrong)))
        self.logger.info('------------------------')
