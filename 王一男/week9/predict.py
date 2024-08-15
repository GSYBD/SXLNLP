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
        self.use_bert = config["use_bert"]
        if self.use_bert:
            self.tokenizer = BertTokenizer.from_pretrained(config['pretrain_model_path'], add_special_tokens=False)
        else:
            self.vocab = self.load_vocab(config["vocab_path"])
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print("模型加载完毕!")

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
            self.config["class_num"] = len(schema)
        return schema

    # 加载字表或词表
    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        self.config["vocab_size"] = len(token_dict)
        return token_dict

    def predict(self, sentence):
        input_id = []
        if self.use_bert:
            encoding = self.tokenizer.encode_plus(sentence, add_special_tokens=False,
                                                  max_length=self.config["max_length"],
                                                  padding='max_length',
                                                  truncation=True)
            input_ids = [encoding['input_ids']]  # 套一层batches
            mask = encoding['attention_mask']
        else:
            for char in sentence:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
            input_ids = [input_id]  # 套一层batches
            mask = []  # 这一分支用不到mask
        with torch.no_grad():
            res = self.model(torch.LongTensor(input_ids), mask)[0]
            if not self.config["use_crf"]:
                res = torch.argmax(res, dim=-1)
        labeled_sentence = ""
        for char, label_index in zip(sentence, res):
            labeled_sentence += char + self.index_to_sign[int(label_index)]
        return labeled_sentence

if __name__ == "__main__":
    sl = SentenceLabel(Config, "model_output/epoch_10.pth")

    sentence = "客厅的颜色比较稳重但不沉重相反很好的表现了欧式的感觉给人高雅的味道"
    res = sl.predict(sentence)
    print(res)

    sentence = "双子座的健康运势也呈上升的趋势但下半月有所回落"
    res = sl.predict(sentence)
    print(res)

