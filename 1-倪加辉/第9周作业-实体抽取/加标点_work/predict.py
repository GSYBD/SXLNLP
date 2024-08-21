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
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_path"])
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))
        self.max_len = config["max_len"]
        self.model.eval()
        print("模型加载完毕!")

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
            self.config["class_num"] = len(schema)
        return schema

    def predict(self, sentence):
        input_id = self.sentence_to_index(sentence)
        with torch.no_grad():
            res = self.model(torch.LongTensor([input_id]))[0]
            res = torch.argmax(res, dim=-1)
        labeled_sentence = ""
        for char, label_index in zip(sentence, res):
            labeled_sentence += char + self.index_to_sign[int(label_index)]
        return labeled_sentence

    def sentence_to_index(self, text):
        input_ids = []
        vocab = self.vocab
        # 使用bert的分词来获取inputId
        if self.config["model_type"] == "bert":
            input_ids = self.tokenizer.encode(text,
                                              padding="max_length",
                                              max_length=self.max_len,
                                              truncation=True)
            return input_ids
        else:
            for char in text:
                input_ids.append(vocab.get(char, vocab['unk']))
        # 填充or裁剪
        input_ids = self.padding(input_ids)
        return input_ids
        # 数据预处理 裁剪or填充

    def padding(self, input_ids, padding_dot=0):
        length = self.config["max_len"]
        padded_input_ids = input_ids
        if len(input_ids) >= length:
            return input_ids[:length]
        else:
            padded_input_ids += [padding_dot] * (length - len(input_ids))
            return padded_input_ids


# 加载字表或词表
def load_vocab(path):
    vocab = {}
    with open(path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            word = line.strip()
            # 0留给padding位置，所以从1开始
            vocab[word] = index + 1
        vocab['unk'] = len(vocab) + 1
    return vocab


if __name__ == "__main__":
    sl = SentenceLabel(Config, "week9_dot.pth")

    sentence = "我真的好想你在每一个雨季下一次见面会更好"
    res = sl.predict(sentence)
    print(res)

    sentence = "天下大势分久必合合久必分"
    res = sl.predict(sentence)
    print(res)
