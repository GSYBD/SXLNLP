import torch
import numpy as np
from collections import defaultdict
from loader import load_data

class Evaluator:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)

    def eval(self, epoch):
        self.model.eval()
        stats_dict = defaultdict(lambda: defaultdict(int))
        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[index * self.config["batch_size"]: (index+1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data
            with torch.no_grad():
                pred_results = self.model(input_ids)

            self.write_stats(labels, pred_results, sentences, stats_dict)

        self.show_stats(stats_dict)

    def write_stats(self, labels, pred_results, sentences, stats_dict):
        pred_results = torch.argmax(pred_results, dim=-1)
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)

            for key in stats_dict.keys():
                stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                stats_dict[key]["样本实体数"] += len(true_entities[key])
                stats_dict[key]["识别出实体数"] += len(pred_entities[key])

    def show_stats(self, stats_dict):
        for key in stats_dict.keys():
            precision = stats_dict[key]["正确识别"] / (stats_dict[key]["识别出实体数"] + 1e-5)
            recall = stats_dict[key]["正确识别"] / (stats_dict[key]["样本实体数"] + 1e-5)
            f1 = 2 * precision * recall / (precision + recall + 1e-5)
            print(f"{key}: Precision={precision}, Recall={recall}, F1={f1}")



