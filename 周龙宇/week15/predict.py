# -*- coding: utf-8 -*-
import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import Config
from model import TorchModel

"""
模型效果测试
"""

class SentenceLabel:
    def __init__(self, config, model_path):
        self.config = config
        self.bio_schema = {"B_object": 0,
                           "I_object": 1,
                           "B_value": 2,
                           "I_value": 3,
                           "O": 4}
        self.attribute_schema = json.load(open(config["schema_path"], encoding="utf8"))
        self.index_to_label = dict((y, x) for x, y in self.attribute_schema.items())
        self.config["bio_count"] = len(self.bio_schema)
        self.config["attribute_count"] = len(self.attribute_schema)
        self.vocab = self.load_vocab(config["vocab_path"])
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("模型加载完毕!")


    # 加载字表或词表
    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        self.config["vocab_size"] = len(token_dict)
        return token_dict

    def decode(self, attribute_label, bio_label, context):
        pred_attribute = self.index_to_label[int(attribute_label)]
        bio_label = "".join([str(i) for i in bio_label.detach().tolist()])
        pred_obj = self.seek_pattern("01*", bio_label, context)
        pred_value = self.seek_pattern("23*", bio_label, context)
        return pred_obj, pred_attribute, pred_value

    def seek_pattern(self, pattern, pred_label, context):
        pred_obj = re.search(pattern, pred_label)
        if pred_obj:
            s, e = pred_obj.span()
            pred_obj = context[s:e]
        else:
            pred_obj = ""
        return pred_obj

    def predict(self, sentence):
        input_id = []
        for char in sentence:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        with torch.no_grad():
            attribute_pred, bio_pred = self.model(torch.LongTensor([input_id]))
            attribute_pred = torch.argmax(attribute_pred)
            bio_pred = torch.argmax(bio_pred[0], dim=-1)
        object, attribute, value = self.decode(attribute_pred, bio_pred, sentence)
        return object, attribute, value

if __name__ == "__main__":
    sl = SentenceLabel(Config, "model_output/epoch_15.pth")

    sentence = "浙江理工大学是一所以工为主，特色鲜明，优势突出，理、工、文、经、管、法、艺术、教育等多学科协调发展的省属重点建设大学。"
    res = sl.predict(sentence)
    print(res)

    sentence = "出露地层的岩石以沉积岩为主（其中最多为碳酸盐岩），在受到乌江的切割下，由内外力共同作用，形成沟壑密布、崎岖复杂的地貌景观。"
    res = sl.predict(sentence)
    print(res)
