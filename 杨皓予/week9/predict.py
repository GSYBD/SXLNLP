# -*- coding: utf-8 -*-
import torch
import re
import json
import numpy as np
from collections import defaultdict

from transformers import BertTokenizer

from config import Config
from model import TorchModel

"""
模型效果测试
"""


class SentenceLabel:
    def __init__(self, config, model_path):
        self.config = config
        self.schema = self.load_schema(config["schema_path"])
        self.index_to_sign = dict((y, x) for x, y in self.schema.items())
        self.tokenizer = load_vocab(config["vocab_path"])
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("模型加载完毕!")

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
            self.config["class_num"] = len(schema)
        return schema

    def encode_sentence(self, review):
        input_id = self.tokenizer.encode(review, max_length=self.config["max_length"],
                                         truncation=True,
                                         padding="max_length")
        return input_id

    def predict(self, sentence):
        input_id = self.tokenizer.encode(sentence)
        with torch.no_grad():
            res = self.model(torch.LongTensor([input_id]))[0]
            res = torch.argmax(res, dim=-1)
        labeled_sentence = ""
        for char, label_index in zip(sentence, res):
            if self.index_to_sign[int(label_index)] == "O":
                labeled_sentence += char
                continue
            labeled_sentence += char + self.index_to_sign[int(label_index)]
        return labeled_sentence


def load_vocab(vocab_path):
    tokenizer = BertTokenizer(vocab_path)
    return tokenizer


if __name__ == "__main__":
    sl = SentenceLabel(Config, r"D:\资料\week9 序列标注问题\week9 序列标注问题\ner\model_output\2epoch_20.pth")

    sentence = "他说:中国政府对目前南亚出现的核军备竞赛的局势深感忧虑和不安。"
    res = sl.predict(sentence)
    print(res)


