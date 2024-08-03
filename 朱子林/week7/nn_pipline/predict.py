# -*- coding: utf-8 -*-

import torch
import os
import time
from model import TorchModel
from loader import encode_predict
'''
模型测试
'''

class Predictor:
    def __init__(self, config):
        self.config = config
        model_path = os.path.join(config["model_path"], "%s_model.pth" % config["model_type"])
        self.out_path = os.path.join(config["model_path"], "predict_%s.txt" % config["model_type"])
        self.model = TorchModel(config)
        # 用GPU训练的
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # 确保模型在正确的设备上
        # 然后加载模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def predict(self, input_ids):
        self.model.eval()
        results = []
        with torch.no_grad():
            start_time = time.perf_counter()
            for input in input_ids:
                encode_input = encode_predict(input, self.config)
                seq = torch.LongTensor([encode_input])
                if torch.cuda.is_available():
                    seq = seq.cuda()
                result = self.model(seq)
                result = torch.argmax(result)
                out_str = input + " " + str(result)
                results.append(out_str)
            end_time = time.perf_counter()
            predict_time = end_time - start_time
            #将预测结果输出到output文件夹下
            with open(self.out_path, mode='w', encoding='utf-8') as file:
                # 遍历数组列表
                for out_w in results:
                    # 写入每一行的数组数据
                    file.write(out_w + '\n')
            return results, predict_time