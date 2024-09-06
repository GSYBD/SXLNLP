# -*- coding: utf-8 -*-
import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import Config
from model import TorchModel
from transformers import BertTokenizer

"""
模型预测
"""

class Predicter:
    def __init__(self, config, model_path):
        self.config = config
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))

    def predict(self, sentence):
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        tokenizer = BertTokenizer.from_pretrained(self.config["bert_path"])

        input_id = tokenizer.encode(sentence, padding='max_length', truncation=True, max_length=self.config['max_length'])

        self.model.eval()
        with torch.no_grad():
            pred_result = self.model(torch.LongTensor([input_id]))  # 不输入labels，使用模型当前参数进行预测

        if not self.config["use_crf"]:
            pred_result = torch.argmax(pred_result, dim=-1)

        if not self.config["use_crf"]:
            pred_result = pred_result.cpu().detach().tolist()

        pred_entities = self.decode(sentence, pred_result)
        print(pred_entities)
        return

    '''
    {
      "B-LOCATION": 0,
      "B-ORGANIZATION": 1,
      "B-PERSON": 2,
      "B-TIME": 3,
      "I-LOCATION": 4,
      "I-ORGANIZATION": 5,
      "I-PERSON": 6,
      "I-TIME": 7,
      "O": 8
    }
    '''
    def decode(self, sentence, labels):
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results

    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

if __name__ == "__main__":
    Config['vocab_size'] = 4622
    p = Predicter(Config, "model_output/epoch_5.pth")

    sentence = "妈妈,你在哪里?杨宗元第一轮比赛的日期目前尚未确定。"
    res = p.predict(sentence)
    print(res)
