# -*- coding: utf-8 -*-
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
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.train_data = load_data(config["train_data_path"], config)
        self.stats_dict = {"correct": 0, "wrong": 0}

    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)

        with torch.no_grad():
            question_matrixs = torch.stack(self.question_ids, dim=0)
            if torch.cuda.is_available():
                question_matrixs = question_matrixs.cuda()
            self.knwb_vectors = self.model.sentence_encoder(question_matrixs)
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)

    def write_stats(self, test_question_vectors, labels):
        similarities = torch.mm(test_question_vectors, self.knwb_vectors.t())
        predicted_indices = torch.argmax(similarities, dim=1)
        for i, predicted_index in enumerate(predicted_indices):
            correct_standard_index = self.question_index_to_standard_question_index[labels[i].item()]
            if correct_standard_index == predicted_index.item():
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1

    def show_stats(self):
        total = self.stats_dict["correct"] + self.stats_dict["wrong"]
        accuracy = self.stats_dict["correct"] / total if total > 0 else 0
        self.logger.info(f"Correct: {self.stats_dict['correct']}, Wrong: {self.stats_dict['wrong']}, Accuracy: {accuracy:.4f}")

    def eval(self, epoch):
        self.logger.info(f"Epoch {epoch}")
        self.stats_dict = {"correct": 0, "wrong": 0}
        self.model.eval()
        self.knwb_to_vector()
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data
            with torch.no_grad():
                test_question_vectors = self.model.sentence_encoder(input_id)
                test_question_vectors = torch.nn.functional.normalize(test_question_vectors, dim=-1)
            self.write_stats(test_question_vectors, labels)
        self.show_stats()


