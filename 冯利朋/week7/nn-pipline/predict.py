"""
预测
"""
import time

import torch.cuda
import torch
from loader import load_data
class Predict:
    def __init__(self, data_path, config, model):
        self.data_path = data_path
        self.config = config
        self.model = model
        self.predict_data = load_data(data_path, config, shuffle=False)
        self.stats_dict = {'correct':0, 'wrong': 0}
    def predict(self):
        start = time.time()
        self.stats_dict = {'correct': 0, 'wrong': 0}
        self.model.eval()
        for index, batch_data in enumerate(self.predict_data):
            if torch.cuda.is_available():
                batch_data = [c.cuda for c in batch_data]
            input_ids, labels = batch_data
            with torch.no_grad():
                pred_res = self.model(input_ids)
            self.write_stats(pred_res, labels)
            break
        end = time.time()
        return self.show_stats(), (end - start)

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
        return correct / (correct + wrong)






