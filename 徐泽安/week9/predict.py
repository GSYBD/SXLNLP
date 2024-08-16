import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import Config
from model import TorchModel
from transformers import BertTokenizer
"""
模型效果测试
"""

class SentenceLabel:
    def __init__(self, config, model_path):
        self.config = config
        self.schema = self.load_schema(config["schema_path"])
        self.index_to_sign = dict((y, x) for x, y in self.schema.items())
        self.vocab = self.load_vocab(config["vocab_path"])
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
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
        # input_id = []
        input_id = self.tokenizer.encode(sentence, max_length=self.config["max_length"], pad_to_max_length=True)
        # for char in sentence:
        #     input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        with torch.no_grad():
            res = self.model(torch.LongTensor([input_id]))
            # print(res)
            # print(type(torch.LongTensor(res)))
            res = torch.argmax(torch.LongTensor(res), dim=-1)
        labeled_sentence = ""
        for char, label_index in zip(sentence, res):
            labeled_sentence += char + self.index_to_sign[int(label_index)]
        return labeled_sentence

if __name__ == "__main__":
    sl = SentenceLabel(Config, "model_output/epoch_20.pth")

    sentence = "日乒名将石川佳纯和张本智和参拜东乡神社，无视日本侵略历史，引中国球迷愤怒。马龙取关石川，体现正义感。运动员应牢记社会责任，历史不容忘却，真相不容扭曲。"
    res = sl.predict(sentence)
    print(res)