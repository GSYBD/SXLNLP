# coding: utf8

import torch
import os
from loader import load_data, encode_predict
from model import TripletLossModel

'''
模型测试
'''

class Predictor:
    def __init__(self, config):
        self.config = config
        self.train_data = load_data(config["train_data_path"], config)
        model_path = os.path.join(config["model_path"], "tripletLossModel.pth")
        self.model = TripletLossModel(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # 确保模型在正确的设备上
        # 然后加载模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

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

    def predict(self, input_ids):
        self.model.eval()
        self.knwb_to_vector()
        results = []
        for input_id in input_ids:
            input_encode = encode_predict(self.config, input_id)
            input_encode = torch.LongTensor([input_encode])
            if torch.cuda.is_available():
                input_encode = input_encode.cuda()
            with torch.no_grad():
                test_q_vectors = self.model(input_encode)
                label, label_id = self.result_to_label(test_q_vectors)
                # 原句， schema，schema id
                result = [input_id, label, label_id]
                results.append(result)
        return results

    def result_to_label(self, test_q_vector):
        # 通 过一次矩阵乘法，计算输入问题和知识库中所有问题的相似度
        # test_q_vector shape [vec_size]   knwb_vectors shape = [n, vec_size]
        res = torch.mm(test_q_vector.unsqueeze(0), self.knwb_vectors.T)
        hit_index = int(torch.argmax(res.squeeze()))  # 命中问题标号
        hit_index = self.q_index_to_standard_q_index[hit_index]  # 转化成标准问编号
        schema_dict = self.train_data.dataset.schema
        for key, value in schema_dict.items():
            if value == hit_index:
                label = key
                break
        #下面那个通过value找Key 不知道为什么不好用了
        # label = schema_dict.get(hit_index, 'UNK')

        return label, hit_index
