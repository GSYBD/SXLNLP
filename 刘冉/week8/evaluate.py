# coding: utf8

import torch
from loader import load_data

"""
模型效果测试
"""


class Evaluator:
    def __init__(self, config, model, train_data,logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, data_type="test", shuffle=False)
        # 由于效果测试需要训练集当做知识库，再次加载训练集。
        self.train_data = train_data
        # 事实上可以通过传参把前面加载的训练集传进来更合理，但是为了主流程代码改动量小，在这里重新加载一遍
        #self.train_data = load_data(config["train_data_path"], config, data_type="train")
        self.stats_dict = {"correct": 0, "wrong": 0}  #用于存储测试结果

    '''
    将知识库中的问题向量化，为匹配做准备
    大概样子:
    standard_q_index:          [13,13，2...]
    question_ids/knwb_vectors  ["宽带坏了","宽带"，"办理业务"...]
    '''

    def knwb_to_vector(self):
        self.q_index_to_standard_q_index = {}
        self.question_ids = []
        for standard_q_index, q_ids in self.train_data.dataset.knwb.items():
            for q_id in q_ids:
                # 记录问题编号到标准问题标号的映射，用来确认答案是否正确
                self.q_index_to_standard_q_index[(len(self.question_ids))] = standard_q_index
                self.question_ids.append(q_id)
        with torch.no_grad():
            q_matrixs = torch.stack(self.question_ids, dim=0)
            if torch.cuda.is_available():
                q_matrixs = q_matrixs.cuda()
            self.knwb_vectors = self.model(q_matrixs)
            # 将所有向量都作归一化 v / |v|
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空前一轮的测试结果
        self.model.eval()
        # 每轮训练的模型参数不一样，生成的向量也不一样，所以需要每轮测试都重新进行向量化
        self.knwb_to_vector()
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data
            with torch.no_grad():
                test_q_vectors = self.model(input_id)
            self.write_stats(test_q_vectors, labels)
        self.show_stats()
        return

    def write_stats(self, test_q_vectors, labels):
        assert len(labels) == len(test_q_vectors)
        for test_q_vector, label in zip(test_q_vectors, labels):
            # 通 过一次矩阵乘法，计算输入问题和知识库中所有问题的相似度
            # test_q_vector shape [vec_size]   knwb_vectors shape = [n, vec_size]
            res = torch.mm(test_q_vector.unsqueeze(0), self.knwb_vectors.T)
            hit_index = int(torch.argmax(res.squeeze()))  # 命中问题标号
            hit_index = self.q_index_to_standard_q_index[hit_index]  # 转化成标准问编号
            if int(hit_index) == int(label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return
