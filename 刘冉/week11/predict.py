# coding: utf-8

import torch
import numpy as np
import os
from config import Config
from model import SFTModel
from transformers import BertTokenizer
import random

class Predictor:
    def __init__(self, config):
        self.config = config
        self.max_length = config["max_length"]
        self.model = SFTModel(config)
        model_path = os.path.join(config["model_path"], "sft_model.pth")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # 确保模型在正确的设备上
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        #token
        self.bertTokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])

    def predict(self, sentence):
        result = self.generate_sentence(sentence, self.model, self.max_length)
        return result

    # 文本生成测试代码
    def generate_sentence(self, openings, model, max_length):
        # reverse_vocab = dict((y, x) for x, y in self.bertTokenizer.vocab.items())
        model.eval()
        with torch.no_grad():
            pred_char = ""
            # 生成了换行符，或生成文本超过200字则终止迭代
            while pred_char != "\n" and len(openings) <= 200:
                openings += pred_char
                x = self.bertTokenizer.encode(openings, add_special_tokens=False, padding='max_length', truncation=True,
                                     max_length=max_length)
                x = torch.LongTensor([x])
                if torch.cuda.is_available():
                    x = x.cuda()
                y = model(x)[0][-1]
                index = self.sampling_strategy(y)
                # pred_char = reverse_vocab[index]
                pred_char = self.bertTokenizer.decode(index)
        return openings

    def sampling_strategy(self, prob_distribution):
        if random.random() > 0.1:
            strategy = "greedy"
        else:
            strategy = "sampling"
        if strategy == "greedy":
            return int(torch.argmax(prob_distribution))
        elif strategy == "sampling":
            prob_distribution = prob_distribution.cpu().numpy()
            return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


if __name__ == "__main__":

    predictor = Predictor(Config)
    sentence = "目击牡丹江部分地税官员的“品质生活"
    res = predictor.predict(sentence)
    print(res)

    sentence = "高中生雇凶杀亲"
    res = predictor.predict(sentence)
    print(res)

    result = predictor.predict("又双叒包揽了")
    print(result)